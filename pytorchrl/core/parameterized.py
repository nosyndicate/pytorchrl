import torch


class Parameterized(object):
    def get_internal_params(self):
        raise NotImplementedError

    def get_internal_named_params(self):
        raise NotImplementedError

    def ordered_params(self):
        """
        Return the named parameters in a list.
        """
        namedparams = sorted(self.get_internal_named_params(), key=lambda x: x[0])
        return [param[1] for param in namedparams]

    def get_param_values(self):
        """
        Get the value of the model parameters in one flat Tensor

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
            have the same format as the return value of get_param_values
            method.
        """
        offset = 0
        for param in self.ordered_params():
            corresponding_param = new_parameters[offset:offset + param.data.numel(
                )].view(param.data.size())
            param.data.copy_(corresponding_param)
            offset += param.data.numel()

    def get_grad_values(self):
        """
        Get the value of the model parameters gradient in one flat Tensor

        Returns
        -------
        gradients (torch.Tensor): A torch Tensor contains the gradient of
            parameters of the module.
        """
        params = self.ordered_params()
        if len(params) > 0:
            # Make the tensors flat and concatenate them
            return torch.cat([param.grad.data.view(-1) for param in params])
        else:
            return torch.zeros((0,))

    def set_grad_values(self, new_gradient):
        """
        Set the value of model parameter gradient using new_gradient.

        Parameters
        ----------
        new_gradient (torch.Tensor): A tensors, this new_gradient should
            have the same format as the return value of get_grad_values
            method.
        """
        offset = 0
        for param in self.ordered_params():
            corresponding_grad = new_gradient[offset:offset + param.grad.numel(
                )].view(param.grad.size())
            param.grad.data.copy_(corresponding_grad)
            offset += param.grad.numel()




