from rascal.representations import SphericalExpansion
from rascal.utils import ClebschGordanReal, compute_lambda_soap, spherical_expansion_reshape

spex = SphericalExpansion(**hypers)

#why is coefficient scaling nescessary? 
feat_scaling = 1e6            # just a scaling to make coefficients O(1)
feats = spex.transform(structures_train).get_features(spex)
ref_feats = feat_scaling*spherical_expansion_reshape(feats, **hypers)

feats_test = spex.transform(structures_test).get_features(spex)
ref_feats_test = feat_scaling*spherical_expansion_reshape(feats_test, **hypers)