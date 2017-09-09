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
