"""ASE calculator comparing SOAP similarity"""

import einops
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from quippy.descriptors import Descriptor


# Writen by Tamas Stenczel
class SoapSimilarityCalculator(Calculator):
    """

    Notes
    -----
    Constraints:
    - single element

    """

    implemented_properties = ["energy", "forces", "energies"]

    def __init__(
        self,
        descriptor: Descriptor,
        ref_soap_vectors: np.ndarray,
        weights=None,
        zeta=1,
        scale=1.0,
        *,
        restart=None,
        label=None,
        atoms=None,
        directory=".",
        **kwargs,
    ):
        """

        Parameters
        ----------
        descriptor
            descriptor calculator object
        ref_soap_vectors
            reference SOAP vectors [n_ref, len_soap]
        weights
            of reference SOAP vectors, equal weight used if not given
        zeta
            exponent of kernel
        scale
            scaling for energy & forces, energy of calculator is
            `-1 * scale * k_soap ^ zeta` where 0 < k_soap < 1
        """
        super().__init__(
            restart=restart,
            label=label,
            atoms=atoms,
            directory=directory,
            **kwargs,
        )

        self.descriptor = descriptor
        self.zeta = zeta
        self.ref_soap_vectors = ref_soap_vectors
        self.scale = scale

        if weights is None:
            self.weights = (
                1 / len(ref_soap_vectors) * np.ones(ref_soap_vectors.shape[0])
            )
        else:
            assert len(weights) == weights.shape[0]
            self.weights = weights

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        # Descriptor calculation w/ gradients
        d_move = self.descriptor.calc(atoms, grad=True)
        print(d_move["data"].shape)
        # -1 * similarity -> 'Energy'
        # k_ab = einops.einsum(
        #     d_move["data"], self.ref_soap_vectors, "a desc, b desc -> a b"
        # ) # very slow with NumPy
        k_ab = d_move["data"] @ self.ref_soap_vectors.T  # [len(atoms), num_ref]
        print(k_ab.shape)
        local_similarity = self.scale * einops.einsum(
            self.weights, k_ab**self.zeta, "b, a b -> a"
        )  # this one is OK, speed about the same as np.dot()
        self.results["energies"] = -1 * local_similarity
        print(local_similarity)
        similarity = np.sum(local_similarity)
        self.results["energy"] = -1 * similarity

        # grad(similarity) -> forces
        # n.b. no -1 since energy is -1 * similarity
        a_cross = d_move["grad_index_0based"]  # [n_cross, 2]
        a_grad_data = d_move["grad_data"]  # [n_cross, 3, len_desc]

        forces = np.zeros(shape=(len(atoms), 3))
        if self.zeta == 1:
            for i_grad, (_, ii) in enumerate(a_cross):
                # forces[ii] += einops.einsum(
                #     self.weights,
                #     a_grad_data[i_grad],
                #     self.ref_soap_vectors,
                #     "bi, cart desc, bi desc -> cart",
                # )
                forces[ii] += np.sum(
                    a_grad_data[i_grad] @ self.ref_soap_vectors.T * self.weights,
                    axis=1,
                )

        else:
            # chain rule - uses k_ab from outside, with z-1 power
            k_ab_zeta = k_ab ** (self.zeta - 1)
            for i_grad, (ci, ii) in enumerate(a_cross):
                # forces[ii] += einops.einsum(
                #     self.weights,
                #     k_ab[ci] ** (self.zeta - 1),
                #     a_grad_data[i_grad],
                #     self.ref_soap_vectors,
                #     "bi, bi, cart desc, bi desc -> cart",
                # ) # this is VERY slow, but easy to understand
                forces[ii] += np.sum(
                    a_grad_data[i_grad]
                    @ self.ref_soap_vectors.T
                    * (self.weights * k_ab_zeta[ci]),
                    axis=1,
                )
            forces *= self.zeta

        self.results["forces"] = self.scale * forces
