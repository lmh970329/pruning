import torch
from torch.nn.utils import prune
from typing import Optional


class MaskedPruning(object):

    def __init__(self, init_candidate_mask: Optional[torch.Tensor], prev_mask: Optional[torch.Tensor]) -> None:
        self.init_candidate_mask = init_candidate_mask
        self.prev_mask = prev_mask

    def get_candidate_mask(self, t: torch.Tensor):

        # First pruning
        if self.prev_mask == None:
            self.prev_mask = torch.ones_like(t)
            
        # No freezed weights
        if self.init_candidate_mask == None:
            self.init_candidate_mask = torch.ones_like(self.prev_mask, dtype=torch.bool)
        
        # Pruning was already applied
        if self.PRUNING_TYPE == 'unstructured':
            slc = (self.prev_mask == 1)

        elif self.PRUNING_TYPE == 'structured':
            if not hasattr(self, 'dim'):
                raise AttributeError(
                    "Pruning methods of PRUNING_TYPE"
                    '"structured" need to have the attribute `dim` defined.'
                )
            n_dims = self.init_candidate_mask.dim()
            dim = self.dim

            if dim < 0:
                dim = n_dims + dim
            if dim < 0:
                raise IndexError(
                    "Index is out of bounds for tensor with dimensions {}".format(
                    n_dims
                    )
                )
            keep_channel = self.prev_mask.sum(dim=[d for d in range(n_dims) if d != dim]) !=0
            slc = [slice(None)] * n_dims
            slc[dim] = keep_channel

        elif self.PRUNING_TYPE == "global":
            n_dims = len(self.init_candidate_mask.shape)
            slc = [slice(None)] * n_dims

        else:
            raise ValueError(
                "Unrecognized PRUNING_TYPE {}".format(self.PRUNING_TYPE)
            )
        
        candidate_mask = self.init_candidate_mask[slc].reshape_as(t)
        return candidate_mask



class L1UnstructuredMasked(MaskedPruning, prune.BasePruningMethod):

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount, init_candidate_mask: Optional[torch.Tensor]=None, prev_mask: Optional[torch.Tensor]=None):
        super(L1UnstructuredMasked, self).__init__(init_candidate_mask, prev_mask)
        self.amount = amount

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):
    
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)

        candidate_mask = self.get_candidate_mask(t)
        max_nparams = candidate_mask.sum()

        if nparams_toprune > max_nparams:
            raise ValueError(
                "amount={} should be smaller than the number of "
                "parameters to prune={}".format(nparams_toprune, max_nparams)
            )

        mask = default_mask.clone(memory_format=torch.contiguous_format)
        
        candidate = t[candidate_mask]
        partial_mask = mask[candidate_mask]

        if nparams_toprune != 0:
            topk = torch.topk(torch.abs(candidate), k=nparams_toprune, largest=False)
            partial_mask[topk.indices] = 0

            mask[candidate_mask] = partial_mask

        return mask
    
    @classmethod
    def apply(cls, module, name, amount, init_candidate_mask=None, prev_mask=None, importance_scores=None):
        return super(L1UnstructuredMasked, cls).apply(
            module,
            name,
            amount=amount,
            init_candidate_mask=init_candidate_mask,
            prev_mask=prev_mask,
            importance_scores=importance_scores
        )



class LnStructuredMasked(MaskedPruning, prune.BasePruningMethod):

    PRUNING_TYPE = "structured"

    def __init__(self, amount, n, dim=-1, init_candidate_mask: Optional[torch.Tensor]=None, prev_mask: Optional[torch.Tensor]=None):
        super(LnStructuredMasked, self).__init__(init_candidate_mask, prev_mask)
        self.amount = amount
        self.n = n
        self.dim = dim

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):

        prune._validate_structured_pruning(t)
        prune._validate_pruning_dim(t, self.dim)

        tensor_size = t.shape[self.dim]
        
        candidate_mask = self.get_candidate_mask(t)

        n_dims = candidate_mask.dim()
        
        keep_channels = candidate_mask.sum([d for d in range(n_dims) if d != self.dim]) != 0
        
        slc = [slice(None)] * n_dims
        slc[self.dim] = keep_channels

        candidate = t[slc]
        candidate_size = candidate.shape[self.dim]

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = candidate_size - nparams_toprune

        prune._validate_pruning_amount(nparams_toprune, candidate_size)

        norm = prune._compute_norm(candidate, self.n, self.dim)
        topk = torch.topk(norm, k=nparams_tokeep, largest=True)

        def make_mask(t, dim, indices):
            mask = torch.zeros_like(t)

            slc = [slice(None)] * len(t.shape)

            slc[dim] = indices
            mask[slc] = 1
            return mask
        
        if nparams_toprune == 0:
            pass
        else:
            mask[slc] = make_mask(candidate, self.dim, topk.indices)
            mask *= default_mask.to(dtype=mask.dtype)

        return mask

    @classmethod
    def apply(cls, module, name, amount, n, dim, init_candidate_mask=None, prev_mask=None, importance_scores=None):
        return super(LnStructuredMasked, cls).apply(
            module,
            name,
            amount=amount,
            n=n,
            dim=dim,
            init_candidate_mask=init_candidate_mask,
            prev_mask=prev_mask,
            importance_scores=importance_scores
        )



class L1UnstructuredSelected(prune.L1Unstructured):

    def __init__(self, amount, candidate_mask=None, mask=None):
        super().__init__(amount)
        self.candidate_mask = candidate_mask
        self.mask = mask


    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):
        if self.candidate_mask is None:
            return super(L1UnstructuredSelected, self).compute_mask(t, default_mask)
        
        if self.mask is not None:
            slc = (self.mask == 1)
            candidate_mask = self.candidate_mask[slc]
        else:
            candidate_mask = self.candidate_mask.reshape(-1)

        slc = (candidate_mask == 1)

        mask = default_mask.clone(memory_format=torch.contiguous_format)
        mask[slc] = super(L1UnstructuredSelected, self).compute_mask(t[slc], default_mask[slc])
        return mask
    

    @classmethod
    def apply(cls, module, name, amount, candidate_mask=None, mask=None, importance_scores=None):
        return super(prune.L1Unstructured, cls).apply(module, name, amount, candidate_mask=candidate_mask, mask=mask, importance_scores=importance_scores)



class L1UnstructuredPartitioned(prune.L1Unstructured):

    def __init__(self, amount, npartitions):
        super(L1UnstructuredPartitioned, self).__init__(amount)
        self.npartitions = npartitions

    def compute_mask(self, t, default_mask):
        t_partitions = torch.split(t, t.shape[0] // self.npartitions, dim=0)
        default_mask_partitions = torch.split(default_mask, default_mask.shape[0] // self.npartitions, dim=0)
        mask_partitions = [super(L1UnstructuredPartitioned, self).compute_mask(t_p, dm_p) for t_p, dm_p in zip(t_partitions, default_mask_partitions)]
        return torch.concat(mask_partitions, dim=0)
    
    @classmethod
    def apply(cls, module, name, amount, npartitions, importance_scores=None):
        return super(prune.L1Unstructured, cls).apply(module, name, amount, npartitions, importance_scores=importance_scores)



class L1UnstructuredInversed(prune.L1Unstructured):

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):
        
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)

        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=True)
            mask.view(-1)[topk.indices] = 0

        return mask
    
    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        return super(prune.L1Unstructured, cls).apply(module, name, amount, importance_scores=importance_scores)



def l1_unstructured_masked(module, name, amount, init_candidate_mask, prev_mask, importance_scores=None):
    L1UnstructuredMasked.apply(
        module=module,
        name=name,
        amount=amount,
        init_candidate_mask=init_candidate_mask,
        prev_mask=prev_mask,
        importance_scores=importance_scores
    )
    return module



def l1_unstructured_partitioned(module, name, amount, npartitions, importance_scores=None):
    L1UnstructuredPartitioned.apply(
        module,
        name,
        amount,
        npartitions,
        importance_scores=importance_scores
    )
    return module



def l1_unstructured_inversed(module, name, amount, importance_scores=None):
    L1UnstructuredInversed.apply(
        module,
        name,
        amount,
        importance_scores=importance_scores
    )
    return module



def ln_structured_masked(module, name, amount, n, dim, init_candidate_mask, prev_mask, importance_scores=None):
    LnStructuredMasked.apply(
        module=module,
        name=name,
        amount=amount,
        n=n,
        dim=dim,
        init_candidate_mask=init_candidate_mask,
        prev_mask=prev_mask,
        importance_scores=importance_scores
    )
    return module



def l1_unstructured_selected(module, name, amount, candidate_mask=None, mask=None, importance_scores=None):
    L1UnstructuredSelected.apply(
        module,
        name,
        amount,
        candidate_mask=candidate_mask,
        mask=mask,
        importance_scores=importance_scores
    )
    return module



def ln_attention_head_structured(parameters, amount, n):

    qkv = []
    head_dims = []
    for module, name, head_dim in parameters:
        weight = getattr(module, name)
        if isinstance(head_dim, int):
            qkv.append(weight)
            head_dims.append(head_dim)
        elif isinstance(head_dim, (list, tuple)):
            n_heads = weight.shape[0] // sum(head_dim)
            for segment in torch.split(weight, [dim * n_heads for dim in head_dim], dim=0):
                qkv.append(segment)
            head_dims += list(head_dim)

    q, k, v = qkv

    n_heads = q.shape[0] // head_dims[0]
    embed_dim = q.shape[-1]

    # Dimension : n_heads * [1, head_dim, embed_dim]
    q_heads = [q_head.unsqueeze(0) for q_head in torch.split(q, head_dims[0], dim=0)]
    k_heads = [k_head.unsqueeze(0) for k_head in torch.split(k, head_dims[1], dim=0)]
    v_heads = [v_head.unsqueeze(0) for v_head in torch.split(v, head_dims[2], dim=0)]

    # Dimension : n_heads, q_dim + k_dim + v_dim, embed_dim
    head_gathered = torch.concat(
        [torch.concat([q_head, k_head, v_head], dim=1) for q_head, k_head, v_head in zip(q_heads, k_heads, v_heads)],
        dim=0
    )

    default_mask = torch.ones_like(head_gathered)
    computed_mask = prune.LnStructured(amount, n, dim=0).compute_mask(head_gathered, default_mask)

    # Dimension : 3 * [head_dim, n_heads, embed_dim]
    masks = torch.permute(computed_mask, (1, 0, 2)).split(head_dims, dim=0)
    qkv_masks = [mask.transpose(0, 1).reshape(-1, embed_dim) for mask in masks]

    idx = 0
    ret = list()
    for module, name, head_dim in parameters:
        mask = None
        if isinstance(head_dim, int):
            mask = qkv_masks[idx]
            idx += 1
        elif isinstance(head_dim, (list, tuple)):
            n = len(head_dim)
            mask = torch.concat(qkv_masks[idx:idx+n], dim=0)
            idx += n
        ret.append(prune.custom_from_mask(module, name, mask=mask))
    
    return ret