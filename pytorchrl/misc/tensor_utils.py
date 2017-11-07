import torch
from torch.autograd import Variable


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def running_average_tensor_list(first_list, second_list, rate):
        """
        Return the result of
        first_list * (1 - rate) + second_list * rate

        Parameter
        ---------
        first_list (list) : A list of pytorch Tensors
        second_list (list) : A list of pytorch Tensors, should have the same
            format as first_list.
        rate (float): a learning rate, in [0, 1]

        Returns
        -------
        results (list): A list of Tensors with computed results.
        """
        results = []
        assert len(first_list) == len(second_list)
        for first_t, second_t in zip(first_list, second_list):
            assert first_t.shape == second_t.shape
            result_tensor = first_t * (1 - rate) + second_t * rate
            results.append(result_tensor)

        return results

def constant(value):
    """
    Return a torch Variable for computation. This is a function to help
    write short code.

    pytorch require multiplication take either two variables
    or two tensors. And it is recommended to wrap a constant
    in a variable which is kind of silly to me.
    https://discuss.pytorch.org/t/adding-a-scalar/218

    Parameters
    ----------
    value (float): The value to be wrapped in Variable

    Returns
    -------
    constant (Variable): The Variable wrapped the value for computation.
    """
    return Variable(torch.Tensor([value])).type(torch.FloatTensor)
