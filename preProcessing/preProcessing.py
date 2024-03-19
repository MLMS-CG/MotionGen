import torch
import numpy as np
import os
import mmap
import json
import utils.welford_means_stds as w
from typing import *
from scipy.spatial.transform import Rotation as R

from os import path
from human_body_prior.body_model.body_model import BodyModel

with open("preProcessing/default_options_dataset.json", "r") as outfile:
    opt = json.load(outfile)

# evecs
print("loading evecs")
with open("data/evecs_4096.bin", "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    evecs = torch.tensor(np.frombuffer(
        mm[:], dtype=np.float32)).view(6890, 4096).to(opt["device"])

evecs = evecs[:, :opt["nb_freqs"]].transpose(0, 1)

# Load body model
print("loading body models")
num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

# male
bm_male = "data/smplh/male/model.npz"
dmpl_male = "data/dmpls/male/model.npz"

smplh_data = np.load(bm_male)
dmpl_data = np.load(dmpl_male)

print("loading male")
bm_male = BodyModel(
    bm_fname=bm_male,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=dmpl_male,
)

print("sending to gpu")
bm_male = bm_male.to(opt["device"])

# female
print("loading female")
bm_female = "data/smplh/female/model.npz"
dmpl_female = "data/dmpls/female/model.npz"

bm_female = BodyModel(
    bm_fname=bm_female,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=dmpl_female,
)

print("sending to gpu")
bm_female = bm_female.to(opt["device"])

os.makedirs(opt["path_dataset"], exist_ok=True)

def selectDataset(babelPath: str, categoryNeeded: Optional[List[int]]) -> Dict[str, List]:
    # read babel label
    files = ["train.json", "val.json", "test.json"]
    labelSets = []
    for labelFile in files:
        f = open(babelPath + labelFile)
        labelSets.append(json.load(f))

    # dict to save information needed
    seqInfos = {} # path: [[action, start, end], [],...]
    for labelSet in labelSets:
        for key in labelSet.keys():
            labelData = labelSet[key]
            frameFlag = 0 # if the sequence has frame level annotation
            # Build the entry in dict
            path = labelData["feat_p"]

            # some may not have frame level annotations
            if labelData["frame_ann"]:
                labels = labelData["frame_ann"]["labels"]
                frameFlag = 1
            else:
                # When there is no frame level annotation, 
                # use the sequence level annotion
                labels = labelData["seq_ann"]["labels"]
                # There should be no more than one sequence level annotion
                # All frames in this sequence will be assigned to this annotion
                if len(labels)>1:
                    continue
            for label in labels:
                startTime = -1
                endTime = -1
                if frameFlag:
                    startTime = label["start_t"]
                    endTime = label["end_t"]
                # some action are null
                if label["act_cat"]:
                    for act in label["act_cat"]:
                        if categoryNeeded and (act in categoryNeeded):
                            if path not in seqInfos.keys():
                                seqInfos[path] = [] 
                            seqInfos[path].append([act, startTime, endTime])
    return seqInfos

    


def fill_dataset(seqInfos):

    print()
    print("Creating dataset ...")
    print()

    print(os.getcwd())

    # compute means and stds
    welford = w.Welford(
        (opt["nb_freqs"], 3),
        opt["path_dataset"]
        + "_means_stds.pt",
        opt["device"]
    )

    total_nb_frames = 0
    total_nb_samples = 0

    path_dataset_file = opt["path_dataset"] + "dataset.bin"
    path_trans_file = opt["path_dataset"] + "trans.bin"
    path_rot_file = opt["path_dataset"] + "rots.bin"
    path_lengths_file = opt["path_dataset"] + "lengths.bin"
    path_tpose_file = opt["path_dataset"] + "tpose.bin"
    actions = []
    # dataset file
    with \
            open(path_dataset_file, "wb") as dataset_file, \
            open(path_lengths_file, "wb") as lengths_file, \
            open(path_trans_file, "wb") as trans_file, \
            open(path_rot_file, "wb") as rots_file, \
            open(path_tpose_file, "wb") as tpose_file:

        for sequence_path in seqInfos.keys():

            npz_data = np.load(
                path.join(opt["amass_directory"], sequence_path)
            )

            subject_gender = npz_data["gender"]
            
            # fps during caption
            mocap_framerate = npz_data["mocap_framerate"]
            length = len(npz_data["poses"])
            timeStamp = np.arange(length) * 1/mocap_framerate
            # index for down sample and for certain time period
            for info in seqInfos[sequence_path]:

                action, startTime, endTime = info
                # select based on time period
                selectedIndex = np.where((timeStamp>=startTime) & (timeStamp<=endTime))[0]
                # down sample
                sampledNum = int((endTime - startTime) * opt["framerate"])
                if sampledNum<=0:
                    continue
                selected = (np.round(np.linspace(selectedIndex[0], selectedIndex[-1], num = sampledNum, endpoint=True))).astype(np.int16)

                if subject_gender == "male":
                    current_bm = bm_male
                elif subject_gender == "female":
                    current_bm = bm_female
                else:
                    print('neutral gender')
                    continue

                new_npz_data = {}

                for k, v in npz_data.items():
                    if k not in [
                        "root_orient",
                        "pose_body",
                        "pose_hand",
                        "trans",
                        "dmpls",
                        "poses",
                    ]:
                        new_npz_data[k] = npz_data[k]
                    else:
                        new_npz_data[k] = npz_data[k][selected]
                        """ if k == "pose_body":
                                    new_npz_data[k][:, :3] = 0 """

                time_length = len(new_npz_data["trans"])
                total_nb_frames += time_length

                # for gpu memory consumption, if the anim is too long
                max_len = 500

                length = 0

                r1 = R.from_rotvec([0.5*np.pi,0,0]).as_matrix()
                r1_reverse = R.from_rotvec([-0.5*np.pi,0,0]).as_matrix()
                body_parms_t = {
                        # controls the body shape
                        "betas": torch.Tensor(
                            np.repeat(
                                new_npz_data["betas"][:num_betas][
                                    np.newaxis
                                ],
                                repeats=2,
                                axis=0,
                            )
                        ).to(opt["device"]),
                    }

                body_tpose = current_bm(**body_parms_t)
                roots_tpose = body_tpose.Jtr[:,0]
                verts_tpose = np.matmul(
                        r1,
                        (body_tpose.v[0]-roots_tpose[0]).cpu().numpy().T
                    ).T
                
                coeffs_tpose = np.matmul(evecs.cpu().numpy(), verts_tpose)
                coeffs_tpose.tofile(tpose_file)

                for i in range(0, len(new_npz_data["trans"][:]), max_len):
                    trans = new_npz_data["trans"][i: i + max_len, :]

                    body_parms = {
                        # "root_orient": torch.Tensor(
                        #     new_npz_data["poses"][i: i + max_len, 0:3]
                        # ).to(opt["device"]),
                        # "trans": torch.Tensor(
                        #     trans
                        # ).to(opt["device"]),
                        # controls the body
                        "pose_body": torch.Tensor(
                            new_npz_data["poses"][i: i + max_len, 3:66]
                        ).to(opt["device"]),
                        # controls the finger articulation
                        "pose_hand": torch.Tensor(
                            new_npz_data["poses"][i: i + max_len, 66:]
                        ).to(opt["device"]),
                        # controls the body shape
                        "betas": torch.Tensor(
                            np.repeat(
                                new_npz_data["betas"][:num_betas][
                                    np.newaxis
                                ],
                                repeats=len(
                                    new_npz_data["trans"][i: i + max_len]
                                ),
                                axis=0,
                            )
                        ).to(opt["device"]),
                    }

                    try:
                        body_pose_beta = current_bm(**body_parms)
                    except Exception as e:
                        print(e)
                        print("sequence_path: ", sequence_path)
                        print(
                            "len: ",
                            new_npz_data["poses"][
                                i: i + max_len, 3:66
                            ].shape[0],
                        )
                    roots = body_pose_beta.Jtr[:,0]


                    verts = np.stack([
                        np.matmul(
                            r1,
                            (body_pose_beta.v[k]-roots[k]).cpu().numpy().T
                        ).T
                        for k in range(len(body_pose_beta.v))
                    ])


                    coeffs = torch.matmul(evecs, torch.tensor(verts).to("cuda").to(torch.float32))

                    welford.aggregate(coeffs.clone(), flatten=False)

                    coeffs.cpu().numpy().tofile(dataset_file)
                    (trans+roots.cpu().numpy()).tofile(trans_file)
                    np.stack([
                        R.from_matrix
                        (
                            np.matmul(
                                R.from_rotvec(mat).as_matrix(), 
                                r1_reverse
                            ) 
                        ).as_rotvec()
                        for mat in new_npz_data["poses"][i: i + max_len, 0:3]
                    ]).tofile(rots_file)
                    length += coeffs.shape[0]

                total_nb_samples += 1

                length = np.array(length, dtype=int)
                length.tofile(lengths_file)
                actions.append(action)

            # uncomment if you want to create few samples for testing
            '''if total_nb_samples > 100:
                if train_test == "train":
                    welford.finalize()
                    welford.save()

                return'''
    np.save(opt["path_dataset"] + "action.npy", actions)
    welford.finalize()
    welford.save()


    print("total_nb_frames", total_nb_frames)
    print("total_nb_samples", total_nb_samples)

    return


def create_dataset():
    selectedData = selectDataset(opt["babel_directory"], opt["action_categories"])

    fill_dataset(selectedData)

    opt_created_dataset = {}
    opt_created_dataset["nb_freqs"] = opt["nb_freqs"]
    opt_created_dataset["framerate"] = opt["framerate"]

    with open(
        opt["path_dataset"] + "infos.json",
        "w",
    ) as outfile:
        json.dump(
            opt_created_dataset,
            outfile,
            sort_keys=True,
            indent=4,
        )
