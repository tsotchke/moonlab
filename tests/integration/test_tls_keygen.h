#ifndef MOONLAB_TEST_TLS_KEYGEN_H
#define MOONLAB_TEST_TLS_KEYGEN_H

#include <openssl/evp.h>
#include <openssl/rsa.h>

/* EVP_RSA_gen was added in OpenSSL 3.0.  Use the EVP key-generation
 * interface available in both OpenSSL 1.1 and 3.x so the TLS integration
 * tests also build on enterprise Linux distributions shipping 1.1.1. */
static EVP_PKEY *moonlab_test_generate_rsa_key(void)
{
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, NULL);
    EVP_PKEY *key = NULL;

    if (ctx == NULL ||
        EVP_PKEY_keygen_init(ctx) <= 0 ||
        EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, 2048) <= 0 ||
        EVP_PKEY_keygen(ctx, &key) <= 0) {
        EVP_PKEY_free(key);
        key = NULL;
    }

    EVP_PKEY_CTX_free(ctx);
    return key;
}

#endif /* MOONLAB_TEST_TLS_KEYGEN_H */
