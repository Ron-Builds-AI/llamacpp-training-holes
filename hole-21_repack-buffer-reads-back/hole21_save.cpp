// hole21_save.cpp - load a GGUF on the CPU and save it again with llama_model_save_to_file.
// usage: hole21_save MODEL.gguf OUT.gguf repack|norepack
// repack leaves llama.cpp's default (weights repacked where the CPU has a layout for them); norepack sets use_extra_bufts = false.
// Prints how many tensors the loader logged as repacked, then saves. Exit 0 means the save returned.
#include "llama.h"

#include <cstdio>
#include <cstring>

static int g_repacked = 0;

static void log_cb(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (strstr(text, "repack tensor") != nullptr) {
        g_repacked++;
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    if (argc != 4 || (strcmp(argv[3], "repack") != 0 && strcmp(argv[3], "norepack") != 0)) {
        fprintf(stderr, "usage: %s MODEL.gguf OUT.gguf repack|norepack\n", argv[0]);
        return 2;
    }
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers    = 0;
    mp.use_extra_bufts = strcmp(argv[3], "repack") == 0;

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        fprintf(stderr, "hole21: load failed\n");
        return 3;
    }
    fprintf(stderr, "hole21: mode %s, tensors logged as repacked: %d\n", argv[3], g_repacked);
    fflush(stderr);

    llama_model_save_to_file(model, argv[2]);
    fprintf(stderr, "hole21: save returned\n");

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
