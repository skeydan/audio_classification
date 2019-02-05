# This code should run with TensorFlow >= 1.13.
# It was designed to run in graph as well as eager modes.
# If running in eager mode only, some pieces of code can be substantially simplified.

library(keras)
#use_implementation("tensorflow")

library(tensorflow)
#tfe_enable_eager_execution(device_policy = "silent")

library(tfdatasets)

library(stringr)
library(dplyr)

use_session_with_seed(7777,
                      disable_gpu = FALSE,
                      disable_parallel_cpu = FALSE)

files <- fs::dir_ls(path = "data_1/speech_commands_v0.01/",
                    recursive = TRUE,
                    glob = "*.wav")

files <- files[!str_detect(files, "background_noise")]

df <- data_frame(
  fname = files,
  class = fname %>%
    str_extract("1/.*/") %>%
    str_replace_all("1/", "") %>%
    str_replace_all("/", ""),
  class_id = class %>% as.factor() %>% as.integer() - 1L
)

batch_size <- 32
buffer_size <- nrow(df)

type = "log_magnitude"

window_size_ms <- 30L 
window_stride_ms <- 10L 
lower_edge_hertz <- 0
upper_edge_hertz <- 2595 * log10(1 + (16000 / 2) / 700)
num_mel_bins <- 64L
num_mfccs <- 13L

data_generator <- function(df,
                           window_size_ms,
                           window_stride_ms) {
  
  # assume sampling rate is the same in all samples
  sampling_rate <-
    tf$audio$decode_wav(tf$read_file(tf$reshape(df$fname[[1]], list()))) %>% .$sample_rate
  
  samples_per_window <- (sampling_rate * window_size_ms) %/% 1000L   # unit = samples
  stride_samples <-  (sampling_rate * window_stride_ms) %/% 1000L   # unit = samples
  
  n_periods <-
    tf$shape(
      tf$range(
        samples_per_window %/% 2L,
        16000L - samples_per_window %/% 2L,
        stride_samples
      )
    )[1] + 1L
  
  n_fft_coefs <-
    (2 ^ tf$ceil(tf$log(
      tf$cast(samples_per_window, tf$float32)
    ) / tf$log(2)) /
      2 + 1L) %>% tf$cast(tf$int32)
  
  ds <- tensor_slices_dataset(df) %>%
    dataset_shuffle(buffer_size = buffer_size)
  
  ds <- ds %>%
    dataset_map(function(obs) {
      wav <-
        tf$audio$decode_wav(tf$read_file(tf$reshape(obs$fname, list())))
      samples <- wav$audio
      samples <- samples %>% tf$transpose(perm = c(1L, 0L))
      
      stft_out <- tf$signal$stft(samples,
                              frame_length = samples_per_window,
                              frame_step = stride_samples)
      
      magnitude_spectrograms <- tf$abs(stft_out)
      
      if (type == "log_magnitude") {
        log_magnitude_spectrograms <- tf$log(magnitude_spectrograms + 1e-6)
 
      } else if (type == "log_mel" || type == "mfcc") {
        # can't use when building static graph as it needs evaluation
        # num_spectrogram_bins <- magnitude_spectrograms$shape[-1]$value 
        linear_to_mel_weight_matrix <-
          tf$signal$linear_to_mel_weight_matrix(
            num_mel_bins,
            #num_spectrogram_bins,
            257L,
            # can't use when building static graph as we need to pass tensor not scalar
            # tf$cast(sampling_rate, tf$float32),
            16000,
            lower_edge_hertz,
            upper_edge_hertz
          )
        mel_spectrograms <- tf$tensordot(magnitude_spectrograms,
                                         linear_to_mel_weight_matrix,
                                         1L)
        log_mel_spectrograms <- tf$log(mel_spectrograms + 1e-6)
        
        if (type == "mfcc") {
          # Keep the first `num_mfccs` MFCCs.
          mfccs <- tf$signal$mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[ , , 1:num_mfccs]
        }

      }
      
      response <- tf$one_hot(obs$class_id, 30L)
      
      if (type == "log_magnitude") {
        # move batch dimension to channels
        input <-
          tf$transpose(log_magnitude_spectrograms, perm = c(1L, 2L, 0L))
        
      } else if (type == "log_mel") {
        input <-
          tf$transpose(log_mel_spectrograms, perm = c(1L, 2L, 0L))
        
      } else if (type == "mfcc") {
        input <-
          tf$transpose(mfccs, perm = c(1L, 2L, 0L))
      }

      list(input, response)
      
    })
  
  n_coefs <-  if (type == "log_magnitude") {
    n_fft_coefs
  } else if (type == "log_mel") {
    tf$constant(num_mel_bins)
  } else if (type == "mfcc") {
    tf$constant(num_mfccs)
  }
  
  ds <- ds %>%
    dataset_repeat()
  ds %>%
    dataset_padded_batch(
      batch_size = batch_size,
      padded_shapes = list(tf$stack(list(
        n_periods, n_coefs,-1L
      )),
      tf$constant(-1L, shape = shape(1L))),
      drop_remainder = TRUE
    )
  
}

id_train <- sample(nrow(df), size = 0.7 * nrow(df))
n_train <- length(id_train)
n_test <- nrow(df) - n_train

ds_train <- data_generator(df[id_train, ],
                           window_size_ms = window_size_ms,
                           window_stride_ms = window_stride_ms)
ds_val <- data_generator(df[-id_train, ],
                         window_size_ms = window_size_ms,
                         window_stride_ms = window_stride_ms)

if (!tf$executing_eagerly()) {
  sess <- tf$Session()
  batch <- next_batch(ds_train)
  str(sess$run(batch))
  n_periods <- sess$run(batch)[[1]] %>% dim() %>% .[2]
  n_coefs <- sess$run(batch)[[1]] %>% dim() %>% .[3]
} else {
  batch <- make_iterator_one_shot(ds_train) %>% iterator_get_next()
  n_periods <- batch[[1]] %>% dim() %>% .[2]
  n_coefs <- batch[[1]] %>% dim() %>% .[3]
}


model <- keras_model_sequential()
model %>%
  layer_conv_2d(
    input_shape = c(n_periods, n_coefs, 1),
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    padding = "same"
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = 'relu',
                padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = 'relu',
                padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 256,
                kernel_size = c(3, 3),
                activation = 'relu',
                padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 30, activation = 'softmax')

model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  ds_train,
  epochs = 5,
  steps_per_epoch = trunc(n_train / batch_size),
  validation_data = ds_val,
  validation_steps = trunc(n_test / batch_size)
)
