import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):

    databatch = [b['inp'] for b in batch]
    tpose = [b['tpose'] for b in batch]
    # lenbatch = [b['lengths'] for b in batch]

    # databatchTensor = collate_tensors(databatch)
    # tposeTensor = collate_tensors(tpose)
    databatchTensor = torch.stack(databatch)
    tposeTensor = torch.stack(tpose).unsqueeze(1)
    # lenbatchTensor = torch.as_tensor(lenbatch)
    # maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[1]) # unqueeze for broadcasting

    motion = databatchTensor
    # cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor, 'tpose': tposeTensor}}
    cond = {'y': {'tpose': tposeTensor}}

    if 'action' in batch[0]:
        actionbatch = [b['action'] for b in batch]
        actioncondbatch = []
        for b in batch:
            if b['action']==-1:
                actioncondbatch.append(0)
            else:
                actioncondbatch.append(1)
        cond['y'].update({'action': torch.tensor(actionbatch).unsqueeze(1)})
        cond['y'].update({'actioncond': torch.tensor(actioncondbatch)})
        cond['y'].update({'shapecond': torch.ones_like(torch.tensor(actioncondbatch))})
    
    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(b[0]).to(torch.float32), 
        'tpose': torch.tensor(b[1]).to(torch.float32),
        'action': b[2], 
    } for b in batch]
    return collate(adapted_batch)