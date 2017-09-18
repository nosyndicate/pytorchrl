import torch


class Parameterized(object):
    def get_internal_params(self):
        raise NotImplementedError

    def get_internal_named_params(self):
        raise NotImplementedError

    # def get_param_values(self):
    #     """
    #     Get the values of model parameters.

    #     Returns
    #     -------
    #     param_tensors (list) : A list contains tensors which is
    #         the parameters of the model
    #     """
    #     param_tensors = [parameter.data for parameter in self.get_internal_params()]

    #     return param_tensors

    # def set_param_values(self, new_param_tensors):
    #     """
    #     Set the value of model parameter using new_param.

    #     Parameters
    #     ----------
    #     param_values (list) : A list of tensors, this parameter should
    #         have the same format as the return value of get_param_values
    #         method.
    #     """
    #     # First check if the dimension match
    #     param_tensors = [parameter.data for parameter in self.get_internal_params()]
    #     assert len(param_tensors) == len(new_param_tensors)
    #     for param, new_param in zip(param_tensors, new_param_tensors):
    #         assert param.shape == new_param.shape

    #     for param, new_param in zip(param_tensors, new_param_tensors):
    #         param.copy_(new_param)

    def ordered_params(self):
        """
        Return the named parameters in a list.
        """
        namedparams = sorted(self.get_internal_named_params(), key=lambda x: x[0])
        return [param[1] for param in namedparams]

    def get_param_values(self):
        """
        Get the value of the model parameters in one numpy array.

        Returns
        -------
        params (torch.Tensor): A torch Tensor contains the parameter of
            the module.
        """
        params = self.ordered_params()
        if len(params) > 0:
            # Make the tensors flat and concatenate them
            return torch.cat([param.data.view(-1) for param in params])
        else:
            return torch.zeros((0,))

    def set_param_values(self, new_parameters):
        """
        Set the value of model parameter using parameters.

        Parameters
        ----------
        new_parameters (torch.Tensor): A tensors, this new_parameters should
            have the same format as the return value of get_flat_params_values
            method.
        """
        offset = 0
        for param in self.ordered_params():
            corresponding_param = new_parameters[offset:offset + param.data.numel(
                )].view(param.data.size())
            param.data.copy_(corresponding_param)
            offset += param.data.numel()
