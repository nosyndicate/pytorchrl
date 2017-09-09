class Parameterized(object):
    def get_internal_params(self):
        raise NotImplementedError

    def get_param_values(self):
        """
        Get the values of model parameters.

        Returns
        -------
        param_tensors (list) : A list contains tensors which is the
            clone of the parameters of the model
        """
        param_tensors = [parameter.data for parameter in self.get_internal_params()]

        return param_tensors

    def set_param_values(self, new_param_tensors):
        """
        Set the value of model parameter using new_param

        Parameters
        ----------
        param_values (list) : A list of tensors, this parameter should
            have the same format as the return value of get_param_values
            method.
        """
        # First check if the dimension match
        param_tensors = [parameter.data for parameter in self.get_internal_params()]
        assert len(param_tensors) == len(new_param_tensors)
        for param, new_param in zip(param_tensors, new_param_tensors):
            assert param.shape == new_param.shape

        for param, new_param in zip(param_tensors, new_param_tensors):
            param.copy_(new_param)
