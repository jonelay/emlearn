
#ifndef EML_GBTREES_H
#define EML_GBTREES_H

#include <string.h>
#include <math.h>
#include <eml_trees.h>

#ifdef __cplusplus
extern "C" {
#endif

// Minimum sum for softmax to prevent division by zero
#define EML_GBTREES_SOFTMAX_MIN_SUM 1e-10f

/** @typedef EmlGradientBoosting
\brief Gradient Boosting model

Wraps an EmlTrees structure with gradient boosting-specific parameters.
Supports binary classification (sigmoid), multi-class (softmax), and regression.
*/
typedef struct _EmlGradientBoosting {
    EmlTrees trees;           // Underlying tree ensemble
    float learning_rate;      // Learning rate for tree contributions
    float initial_value;      // Initial prediction (log-odds for binary, constant for regression)
    const float *initial_values; // Per-class initial values (multi-class only, NULL otherwise)
    int8_t n_classes;         // Number of classes (0 for regression)
} EmlGradientBoosting;

/**
* \brief Sigmoid activation function
*/
static inline float
eml_gbtrees_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/**
* \brief Run gradient boosting regression
*
* \param model EmlGradientBoosting instance
* \param features Input data values
* \param features_length Length of input data
*
* \return The predicted value, or NAN on error
*/
float
eml_gbtrees_regress(const EmlGradientBoosting *model,
        const int16_t *features, int8_t features_length)
{
    if (!model || !features) {
        return NAN;
    }

    const EmlTrees *trees = &model->trees;

    if (features_length != trees->n_features) {
        return NAN;
    }
    if (trees->leaf_bits != 32) {
        return NAN;
    }

    const int leaf_size = 4;
    float sum = 0.0f;

    for (int32_t i = 0; i < trees->n_trees; i++) {
        const int32_t leaf_number = eml_trees_predict_tree(trees, trees->tree_roots[i], features, features_length);
        const int32_t leaf_offset = leaf_number * leaf_size;
        float leaf_val;
        memcpy(&leaf_val, trees->leaves + leaf_offset, sizeof(float));
        sum += leaf_val;
    }

    return model->initial_value + model->learning_rate * sum;
}

/**
* \brief Run gradient boosting binary classification probability
*
* \param model EmlGradientBoosting instance
* \param features Input data values
* \param features_length Length of input data
* \param out Buffer to store probabilities [p_class0, p_class1]
* \param out_length Length of output buffer (must be 2)
*
* \return EmlOk on success, or error on failure
*/
EmlError
eml_gbtrees_predict_proba_binary(const EmlGradientBoosting *model,
        const int16_t *features, int8_t features_length,
        float *out, int32_t out_length)
{
    EML_PRECONDITION(features, EmlUninitialized);
    EML_PRECONDITION(out, EmlUninitialized);
    EML_PRECONDITION(out_length == 2, EmlSizeMismatch);

    const EmlTrees *trees = &model->trees;
    EML_PRECONDITION(features_length == trees->n_features, EmlSizeMismatch);

    if (trees->leaf_bits != 32) {
        return EmlUnsupported;
    }

    const int leaf_size = 4;
    float sum = 0.0f;

    for (int32_t i = 0; i < trees->n_trees; i++) {
        const int32_t leaf_number = eml_trees_predict_tree(trees, trees->tree_roots[i], features, features_length);
        const int32_t leaf_offset = leaf_number * leaf_size;
        float leaf_val;
        memcpy(&leaf_val, trees->leaves + leaf_offset, sizeof(float));
        sum += leaf_val;
    }

    const float score = model->initial_value + model->learning_rate * sum;
    const float prob_class1 = eml_gbtrees_sigmoid(score);
    out[1] = prob_class1;
    out[0] = 1.0f - prob_class1;

    return EmlOk;
}

/**
* \brief Run gradient boosting multi-class classification probability
*
* \param model EmlGradientBoosting instance
* \param features Input data values
* \param features_length Length of input data
* \param out Buffer to store probabilities
* \param out_length Length of output buffer (must be n_classes)
*
* \return EmlOk on success, or error on failure
*/
EmlError
eml_gbtrees_predict_proba_multiclass(const EmlGradientBoosting *model,
        const int16_t *features, int8_t features_length,
        float *out, int32_t out_length)
{
    EML_PRECONDITION(features, EmlUninitialized);
    EML_PRECONDITION(out, EmlUninitialized);

    const int n_classes = model->n_classes;
    EML_PRECONDITION(n_classes >= 2, EmlSizeMismatch);
    EML_PRECONDITION(out_length == n_classes, EmlSizeMismatch);
    EML_PRECONDITION(n_classes <= EMTREES_MAX_CLASSES, EmlSizeMismatch);

    const EmlTrees *trees = &model->trees;
    EML_PRECONDITION(features_length == trees->n_features, EmlSizeMismatch);

    // Validate tree count is divisible by n_classes
    EML_PRECONDITION(trees->n_trees % n_classes == 0, EmlSizeMismatch);

    if (trees->leaf_bits != 32) {
        return EmlUnsupported;
    }

    const int leaf_size = 4;
    const int trees_per_iteration = n_classes;
    const int n_iterations = trees->n_trees / trees_per_iteration;

    // Initialize scores with initial values
    for (int k = 0; k < n_classes; k++) {
        out[k] = (model->initial_values != NULL) ? model->initial_values[k] : 0.0f;
    }

    // Accumulate tree predictions
    for (int iter = 0; iter < n_iterations; iter++) {
        for (int k = 0; k < n_classes; k++) {
            const int tree_idx = iter * n_classes + k;
            const int32_t leaf_number = eml_trees_predict_tree(trees, trees->tree_roots[tree_idx], features, features_length);
            const int32_t leaf_offset = leaf_number * leaf_size;
            float leaf_val;
            memcpy(&leaf_val, trees->leaves + leaf_offset, sizeof(float));
            out[k] += model->learning_rate * leaf_val;
        }
    }

    // Apply softmax
    float max_score = out[0];
    for (int k = 1; k < n_classes; k++) {
        if (out[k] > max_score) max_score = out[k];
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < n_classes; k++) {
        out[k] = expf(out[k] - max_score);
        sum_exp += out[k];
    }

    if (sum_exp < EML_GBTREES_SOFTMAX_MIN_SUM) sum_exp = EML_GBTREES_SOFTMAX_MIN_SUM;

    for (int k = 0; k < n_classes; k++) {
        out[k] /= sum_exp;
    }

    return EmlOk;
}

/**
* \brief Run gradient boosting classification probability
*
* \param model EmlGradientBoosting instance
* \param features Input data values
* \param features_length Length of input data
* \param out Buffer to store probabilities
* \param out_length Length of output buffer
*
* \return EmlOk on success, or error on failure
*/
EmlError
eml_gbtrees_predict_proba(const EmlGradientBoosting *model,
        const int16_t *features, int8_t features_length,
        float *out, int32_t out_length)
{
    EML_PRECONDITION(model, EmlUninitialized);

    if (model->n_classes == 2) {
        return eml_gbtrees_predict_proba_binary(model, features, features_length, out, out_length);
    } else {
        return eml_gbtrees_predict_proba_multiclass(model, features, features_length, out, out_length);
    }
}

/**
* \brief Run gradient boosting classification
*
* For binary classification, skips sigmoid computation by comparing
* raw logit score to 0 (equivalent to sigmoid(x) >= 0.5).
*
* \param model EmlGradientBoosting instance
* \param features Input data values
* \param features_length Length of input data
*
* \return The predicted class, or negative error code on failure
*/
int32_t
eml_gbtrees_predict(const EmlGradientBoosting *model,
        const int16_t *features, int8_t features_length)
{
    if (!model || !features) {
        return -EmlTreesErrorLength;
    }

    const int n_classes = model->n_classes;

    if (n_classes == 2) {
        // Binary: skip sigmoid, compare raw score to 0
        // Raw score computation is identical to regression
        const float score = eml_gbtrees_regress(model, features, features_length);
        if (isnan(score)) {
            return -EmlTreesErrorLength;
        }
        return (score >= 0.0f) ? 1 : 0;
    }

    if (n_classes > EMTREES_MAX_CLASSES) {
        return -EmlTreesErrorLength;
    }

    float proba[EMTREES_MAX_CLASSES];
    const EmlError err = eml_gbtrees_predict_proba(model, features, features_length, proba, n_classes);
    if (err != EmlOk) {
        return -EmlTreesUnknownError;
    }

    int32_t best_class = 0;
    float best_prob = proba[0];
    for (int k = 1; k < n_classes; k++) {
        if (proba[k] > best_prob) {
            best_prob = proba[k];
            best_class = k;
        }
    }

    return best_class;
}

#ifdef __cplusplus
}
#endif

#endif // EML_GBTREES_H
