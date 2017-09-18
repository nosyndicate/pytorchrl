
class Distribution(object):


    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of two distributions
        """
        raise NotImplementedError

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two distributions
        """
        raise NotImplementedError

    def likelihood_ratio(self, other, a):
        """
        Compute p_self(a) / p_other(a)

        Parameters
        ----------
        other (Distribution):
        a (Variable):

        Returns
        -------
        ratio (Variable):
        """
        logli = self.log_likelihood(a)
        other_logli = other.log_likelihood(a)
        return (logli - other_logli).exp()

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        raise NotImplementedError

    def entropy(self, dist_info):
        raise NotImplementedError

    def log_likelihood_sym(self, x_var, dist_info_vars):
        raise NotImplementedError

    def likelihood_sym(self, x_var, dist_info_vars):
        raise NotImplementedError

    def log_likelihood(self, a):
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        raise NotImplementedError