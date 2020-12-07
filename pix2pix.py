import tensorflow as tf

import os
import time
import numpy as np
import argparse

from matplotlib import pyplot as plt

class ImageDataLoader:
  def __init__(self, batch_size=8, buffer_size=400, img_dim=256):
    self._feature_description = {
      "height": tf.io.FixedLenFeature([], tf.int64),
      "width": tf.io.FixedLenFeature([], tf.int64),
      "real_image": tf.io.FixedLenFeature([], tf.string),
      "input_image": tf.io.FixedLenFeature([], tf.string)
    }
    self._batch_size = batch_size
    self._buffer_size = buffer_size
    self._img_dim = img_dim

  
  def load_datasets(self, train_edges, test_edges):
    self.train_ds = tf.data.TFRecordDataset(train_edges)


    self.train_ds = tf.data.Dataset.concatenate(tf.data.TFRecordDataset("output_edges/train/fake.tfrecords"), self.train_ds)
    self.train_ds = tf.data.Dataset.concatenate(tf.data.TFRecordDataset("output_edges/train/face_data.tfrecords"), self.train_ds)
    self.train_ds = tf.data.Dataset.concatenate(tf.data.TFRecordDataset(test_edges), self.train_ds)
    
    self.train_ds = self.train_ds.map(self.load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)  
    self.train_ds = self.train_ds.shuffle(self._buffer_size)
    self.train_ds = self.train_ds.batch(self._batch_size)

    self.test_ds = tf.data.TFRecordDataset(test_edges)
    self.test_ds = self.test_ds.map(self.load_image_test)
    self.test_ds = self.test_ds.shuffle(self._buffer_size).batch(self._batch_size)


  def load(self, example_proto):
    example = tf.io.parse_single_example(example_proto, self._feature_description)
    
    # Decode bytes
    real_image = tf.io.decode_raw(example["real_image"], tf.uint8)
    input_image = tf.io.decode_raw(example["input_image"], tf.uint8)

    # Reshape to original images dimensions
    real_image = tf.reshape(real_image, (example["width"], example["height"], 3))
    input_image = tf.reshape(input_image, (self._img_dim , self._img_dim , 3))

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

  def resize(self, input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

  def random_crop(self, input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, self._img_dim, self._img_dim, 3])

    return cropped_image[0], cropped_image[1]

  # normalizing the images to [-1, 1]
  def normalize(self, input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

  @tf.function()
  def random_jitter(self, input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = self.resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = self.random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
      # random mirroring
      input_image = tf.image.flip_left_right(input_image)
      real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

  def load_image_train(self, image_file):
    input_image, real_image = self.load(image_file)
    input_image, real_image = self.random_jitter(input_image, real_image)
    input_image, real_image = self.normalize(input_image, real_image)
    input_image, real_image = self.load(image_file)
    input_image, real_image = self.resize(input_image, real_image,
                                    self._img_dim, self._img_dim)
    input_image, real_image = self.normalize(input_image, real_image)

    return input_image, real_image


  def load_image_test(self, image_file):
    input_image, real_image = self.load(image_file)
    input_image, real_image = self.resize(input_image, real_image,
                                    self._img_dim, self._img_dim)
    input_image, real_image = self.normalize(input_image, real_image)

    return input_image, real_image


class Pix2Pix:
  def __init__(self, output_channels=3, restore=False):

    self._output_channels = 3
    
    if restore:
      # Create checkpoint manager
      self._generator = self.Generator()
      self._discriminator = self.Discriminator()
      self._generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
      self._discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
      
      self._checkpoint_dir = './training_checkpoints'
      self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
      self._checkpoint = tf.train.Checkpoint(generator_optimizer=self._generator_optimizer,
                                      discriminator_optimizer=self._discriminator_optimizer,
                                      generator=self._generator,
                                      discriminator=self._discriminator)
      self._manager = tf.train.CheckpointManager(self._checkpoint, self._checkpoint_dir, max_to_keep=3)
      self._checkpoint.restore(self._manager.latest_checkpoint)


  def Discriminator(self):
    """PatchGAN Discriminator"""
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    # Concatenates the  gan input and real or generated image
    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = self.downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = self.downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = self.downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
    
  def Generator(self):
    """U-Net Pix2Pix Generator."""
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
      self.downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
      self.downsample(128, 4), # (bs, 64, 64, 128)
      self.downsample(256, 4), # (bs, 32, 32, 256)
      self.downsample(512, 4), # (bs, 16, 16, 512)
      self.downsample(512, 4), # (bs, 8, 8, 512)
      self.downsample(512, 4), # (bs, 4, 4, 512)
      self.downsample(512, 4), # (bs, 2, 2, 512)
      self.downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
      self.upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
      self.upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
      self.upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
      self.upsample(512, 4), # (bs, 16, 16, 1024)
      self.upsample(256, 4), # (bs, 32, 32, 512)
      self.upsample(128, 4), # (bs, 64, 64, 256)
      self.upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(self._output_channels, 4,
                                          strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

  def downsample(self, filters, size, apply_batchnorm=True):
    """Downsample block used in U-Net."""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                              kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.LeakyReLU())

    return result

  def upsample(self, filters, size, apply_dropout=False):
    """Upsample block used in U-Net."""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class Trainer:
  def __init__(self, args):
    self._args = args

    self._pix2pix = Pix2Pix()
    self._generator = self._pix2pix.Generator()
    self._discriminator = self._pix2pix.Discriminator()


    self._generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self._discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
    self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # Create checkpoint manager
    self._checkpoint_dir = './training_checkpoints'
    self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
    self._checkpoint = tf.train.Checkpoint(generator_optimizer=self._generator_optimizer,
                                    discriminator_optimizer=self._discriminator_optimizer,
                                    generator=self._generator,
                                    discriminator=self._discriminator)
    self._manager = tf.train.CheckpointManager(self._checkpoint, self._checkpoint_dir, max_to_keep=3)
    self._checkpoint.restore(self._manager.latest_checkpoint)

    self._data_loader = ImageDataLoader(self._args.batch_size, self._args.buffer_size)
    self._data_loader.load_datasets(self._args.train_edges, self._args.test_edges)

    # Create logging summary writer 
    import datetime
    log_dir="logs/"
    self._summary_writer = tf.summary.create_file_writer(
      log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


  def fit(self):
    for epoch in range(self._args.epochs):
      start = time.time()

      #display.clear_output(wait=True)
      if (epoch + 1) % 1 ==0 : 
        for example_input, example_target in self._data_loader.test_ds.take(1):
          self.generate_images(example_input, example_target, epoch)
      
      print("Epoch: ", epoch)

      # Train
      avg_loss = np.zeros(4)
      total = 0
      num_ex = 0
      for n, (input_image, target) in self._data_loader.train_ds.enumerate():
        print('.', end='')
        if (n+1) % 100 == 0:
          print()
        if (n+1) % 100 == 0:
          self._manager.save()
          self.generate_images(example_input, example_target, epoch)
          
        
        num_ex += input_image.shape[0]
        total += 1
        losses = self.train_step(input_image, target, epoch)
        avg_loss += losses
      print()
      print("num_ex", num_ex)
      with self._summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', avg_loss[0]/total, step=epoch)
        tf.summary.scalar('gen_gan_loss', avg_loss[1]/total, step=epoch)
        tf.summary.scalar('gen_l1_loss', avg_loss[2]/total, step=epoch)
        tf.summary.scalar('disc_loss', avg_loss[3]/total, step=epoch)

      # saving (checkpoint) the model every 20 epochs
      if (epoch + 1) % 1 == 0:
          self._manager.save()

      print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                          time.time()-start))
    self._manager.save()

  @tf.function
  def train_step(self, input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = self._generator(input_image, training=True)

      disc_real_output = self._discriminator([input_image, target], training=True)
      disc_generated_output = self._discriminator([input_image, gen_output], training=True)

      gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
      disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
    
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                        self._generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                self._discriminator.trainable_variables)

    self._generator_optimizer.apply_gradients(zip(generator_gradients,
                                            self._generator.trainable_variables))
    self._discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                self._discriminator.trainable_variables))
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss
  
  def generator_loss(self, disc_generated_output, gen_output, target):
    gan_loss = self._binary_crossentropy(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (self._args.gen_lam * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

  def discriminator_loss(self, disc_real_output, disc_generated_output):
    true_real_out = np.full(disc_real_output.shape, 0.9)
    real_loss = self._binary_crossentropy(true_real_out, disc_real_output)

    generated_loss = self._binary_crossentropy(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

  def generate_images(self, test_input, tar, epoch=0):
    prediction = self._generator(test_input, training=True)
    #fig, axs = plt.subplots(4,3, gridspec_kw={'wspace':0, 'hspace':0}, squeeze=True)
    fig, axs = plt.subplots(4,3)
    #plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    # plt.grid("off")
    # plt.grid(b=None)
    for i in range(4):
      axs[i][0].imshow(test_input[i] * 0.5 + 0.5)
      #axs[i][0].set_title(title[0])

      axs[i][0].axis("off")
      axs[i][1].imshow(tar[i] * 0.5 + 0.5)
      #axs[i][1].set_title(title[1])

      axs[i][1].axis("off")
      axs[i][2].imshow(prediction[i] * 0.5 + 0.5)
      #axs[i][2].set_title(title[2])

      axs[i][2].axis("off")
      #plt.subplot(1, 3, i+1)
      #plt.title(title[i])
      # getting the pixel values between [0, 1] to plot it.
      #plt.imshow(display_list[i] * 0.5 + 0.5)
    #plt.subplots_adjust(wspace=0, hspace=0)
    #plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    #plt.show()
    plt.savefig('train_imgs/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def main(args):
  trainer = Trainer(args)
  trainer.fit()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type=int, default=256,
      help="Number of epochs to train.")
  parser.add_argument("--batch_size", type=int, default=8,
      help="Training image batch size.")
  parser.add_argument("--buffer_size", type=int, default=400,
      help="Shuffle buffer size.")
  parser.add_argument("--img_dim", type=int, default=256,
      help="The width/height of output image.")
  parser.add_argument("--output_channels", type=int, default=3,
      help="Output channels of generator.")
  parser.add_argument("--gen_lam", type=int, default=100,
      help="Lambda value to use for generator loss.")
  parser.add_argument("--train_edges", default="output_edges/train/edges.tfrecords",
      help="Training image/edges pairs.")
  parser.add_argument("--test_edges", default="output_edges/test/animeface_edges.tfrecords",
      help="Training image/edges pairs.")
  main(parser.parse_args())
  