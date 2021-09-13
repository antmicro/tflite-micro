/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstring>

#include <zephyr.h>
#include <device.h>
#include <drivers/i2s.h>

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

// The following part defines units that will be used. Most of them is defined
// in the pattern AUDIO_X_Y and means how many Y's every X consists of.
// In the case of LiteX+VexRiscv changing the first 3 of these constants might
// require changes in the design for the i2s driver to start properly.
#define AUDIO_CHANNELS      1
#define AUDIO_FREQ          16000

#define AUDIO_SAMPLE_BYTES  2
#define AUDIO_SAMPLE_BITS   (AUDIO_SAMPLE_BYTES*8)

// Frames consist of values from all channels sampled in a single moment
#define AUDIO_FRAME_SAMPLES AUDIO_CHANNELS
#define AUDIO_FRAME_BYTES   (AUDIO_FRAME_SAMPLES*AUDIO_SAMPLE_BYTES)

// The audio data is received from the driver in blocks of frames
#define AUDIO_BLOCK_FRAMES  128
#define AUDIO_BLOCK_SAMPLES (AUDIO_BLOCK_FRAMES*AUDIO_FRAME_SAMPLES)
#define AUDIO_BLOCK_BYTES   (AUDIO_BLOCK_FRAMES*AUDIO_FRAME_BYTES)
#define AUDIO_BLOCK_MSECS   (AUDIO_BLOCK_FRAMES/(AUDIO_FREQ/1000))

// The size of the memory block given to the driver to store audio data
#define AUDIO_BUF_BLOCKS    128
#define AUDIO_BUF_BYTES     (AUDIO_BUF_BLOCKS*AUDIO_BLOCK_BYTES)

// The size of the ring buffer in which data read from the driver is stored
#define AUDIO_RING_SLOTS    64
#define AUDIO_RING_SAMPLES  (AUDIO_RING_SLOTS*AUDIO_BLOCK_SAMPLES)

// This aligns higher priority to audio gathering than data processing,
// the reason is that the gatherer sleeps most of the time, awaiting new data
#define AUDIO_THREAD_PRIORITY 0
#define MAIN_THREAD_PRIORITY 1

#define RETURN_WITH_ERROR(msg, id) \
  do { \
    TF_LITE_REPORT_ERROR(error_reporter, "Error: %s:%d: " msg "\n", \
      __FILE__, __LINE__, (id)); \
    return kTfLiteError; \
  } while (0);

namespace {
volatile int32_t g_latest_audio_timestamp = 0;
bool g_is_audio_initialized = false;
int16_t g_audio_ring_buffer[AUDIO_RING_SAMPLES];
int16_t g_audio_output_buffer[kMaxAudioSampleSize];

// The i2s driver-specific globals
const struct device *i2s_rx_device;
i2s_config i2s_rx_config;
char i2s_rx_buffer[AUDIO_BUF_BYTES];
k_mem_slab i2s_rx_mem_slab;

// The threading-specific globals
K_THREAD_STACK_DEFINE(thread_audio_stack, 256);
struct k_thread thread_audio;
k_tid_t thread_audio_tid;
} // namespace

// Gather audio data as a thread (not supposed to ever return)
void GatherAudioData(void *p1, void *p2, void *p3) {
  static size_t size;
  static void *block;

  for (int i = 0; ; i = (i+1)%AUDIO_RING_SLOTS) {
    // Read the next block
    i2s_read(i2s_rx_device, &block, &size);
    // Save the block data in the ring buffer
    memcpy((void*)&g_audio_ring_buffer[i*AUDIO_BLOCK_SAMPLES],
      block,
      AUDIO_BLOCK_BYTES);
    // Increase the timestamp by the msec length of a single block
    g_latest_audio_timestamp += AUDIO_BLOCK_MSECS;
    // Delete the block from the internal i2s buffer
    k_mem_slab_free(&i2s_rx_mem_slab, &block);
  }
}

// Initialize the Zephyr i2s driver
TfLiteStatus InitializeAudioProvider(tflite::ErrorReporter *error_reporter) {
  int ret;

  // Obtain i2s device binding
  i2s_rx_device = device_get_binding("i2s_rx");
  if (!i2s_rx_device)
    RETURN_WITH_ERROR("unable to bind i2s_rx device", 0);

  // Initialize memory slab for i2s configuration
  k_mem_slab_init(&i2s_rx_mem_slab,
    i2s_rx_buffer,
    AUDIO_BLOCK_BYTES,
    AUDIO_BUF_BLOCKS);

  // Configure i2s
  i2s_rx_config.word_size = AUDIO_SAMPLE_BITS;
  i2s_rx_config.channels = AUDIO_CHANNELS;
  i2s_rx_config.format = I2S_FMT_DATA_FORMAT_I2S;
  i2s_rx_config.options = I2S_OPT_FRAME_CLK_SLAVE;
  i2s_rx_config.frame_clk_freq = AUDIO_FREQ;
  i2s_rx_config.block_size = AUDIO_BLOCK_BYTES;
  i2s_rx_config.mem_slab = &i2s_rx_mem_slab;
  i2s_rx_config.timeout = -1;

  // Apply the configuration
  ret = i2s_configure(i2s_rx_device, I2S_DIR_RX, &i2s_rx_config);
  if (ret != 0)
    RETURN_WITH_ERROR("i2s_configure failed with error: %d", ret);

  // Start the the device
  ret = i2s_trigger(i2s_rx_device, I2S_DIR_RX, I2S_TRIGGER_START);
  if (ret != 0)
    RETURN_WITH_ERROR("i2s_trigger failed with error: %d", ret);

  return kTfLiteOk;
}

// Get audio samples from the specified moment in time
TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  // On the first run initialize the driver and await the first samples
  if (!g_is_audio_initialized) {
    // Zero the buffers, just in case
    memset(i2s_rx_buffer, 0, sizeof(i2s_rx_buffer));
    memset(g_audio_output_buffer, 0, sizeof(g_audio_output_buffer));

    // Initialize the i2s driver
    if (InitializeAudioProvider(error_reporter) != kTfLiteOk)
      return kTfLiteError;

    // Run the audio gatherer thread
    thread_audio_tid = k_thread_create(&thread_audio,
      thread_audio_stack,
      K_THREAD_STACK_SIZEOF(thread_audio_stack),
      GatherAudioData,
      NULL, NULL, NULL,
      AUDIO_THREAD_PRIORITY,
      0, K_NO_WAIT);

    // Lower the main thread priority to let the audio gatherer thread
    // work until it's blocked
    auto thread_main = k_current_get();
    k_thread_priority_set(thread_main, MAIN_THREAD_PRIORITY);

    // Wait for the first samples to be read
    while (g_latest_audio_timestamp == 0)
      ;

    g_is_audio_initialized = true;
  }

  // Convert the requested times to the corresponding values in the ring buffer
  const int buffer_position = start_ms * (AUDIO_FREQ/1000);
  const int needed_samples = duration_ms * (AUDIO_FREQ/1000);

  // Rewrite the requested samples to the output buffer
  for (int i = 0; i < needed_samples; i++) {
    g_audio_output_buffer[i] =
      g_audio_ring_buffer[(buffer_position + i) % AUDIO_RING_SAMPLES];
  }
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;

  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() {
  return g_latest_audio_timestamp;
}
