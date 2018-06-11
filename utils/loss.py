import pdb
import torch


def margin_loss(v_out, targ, m_p = 0.9, m_m = 0.1, lamb=0.5):
    """
    Implements the margin loss 
    """
    batch_size = v_out.size(0)
    # probabilty values of each prediction in the activity vector v (||v||), is of shape [batch_size]
    v_norm = torch.sqrt(torch.sum(v_out**2, dim=-1)) # v_norm 
    
    ## i) compute left of L_k (L_pos): loss from present labels. 
    # present label prediction probabilities of the capsule net.
    v_pos = v_norm[range(batch_size), targ]
    # m_p-||v_k+||
    v_pos = (m_p - v_pos)
    v_pos[v_pos < 0] = 0
    L_pos = (v_pos**2).sum()

    ## ii) compute left of L_k (L_neg): loss from absent labels.
    # absent label prediction probabilities of the capsule net.
    aux = v_norm.clone()
    # make present label probabilities 0 to disregard those in the computation of loss of absent labels.
    aux[range(aux.size(0)), targ] = 0 
    aux = aux - m_m
    aux[aux<0] = 0
    L_neg = (aux ** 2).sum()   
    
    return L_pos + lamb * L_neg


def margin_loss_v2(v_out, targ, m_p = 0.9, m_m = 0.1, lamb=0.5):    
    """
    Implements the margin loss (tensor notation), but slower.
    """
    batch_size = v_out.size(0)
    # probability of v
    v_norm = torch.sqrt(torch.sum(v_out**2, dim=-1))

    # compute m_plus and m_minus tensors (present labels=m_plus, absent labels=m_minus)
    m_tens = torch.ones(v_norm.shape) * m_m
    m_tens[range(batch_size), targ] = m_p
    # compute lambda tensor (down-weighting absent labels)
    lamb_tens = torch.ones(v_norm.shape) * lamb     # make absent label lambda = lamb(0.5).
    lamb_tens[range(batch_size), targ] = 1  # make present label lambda = 1.
    
    # ||v|| - m : m contains m_p and m_m
    v_norm = v_norm - m_tens
    # change the present label sign (m_p)
    v_norm[range(batch_size), targ] -= 1
    # make non-negative
    v_norm[v_norm<0] = 0
    # compute the margin loss
    return ((v_norm**2)*lamb_tens).sum()

def recons_loss(im_out, im_targ):
    """
    Implements the reconstruction loss for Decoder layer. 
    """
    # im_out is of shape : [batch_size, im_height*im_width]
    # im_targ is of shape : [batch_size, 1, im_height, im_width]
    batch_size = im_targ.size(0)
    im_targ = im_targ.squeeze(1).view(batch_size, -1)
    return ((im_targ - im_out)**2).sum()

