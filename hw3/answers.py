r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I. SCENE"
    temperature = .5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the data because we want to limit the BPTT process. The whole text is huge, and this means that backwards
through time will take too long to calculate. Batches, with detaching of hidden state between them, allow us to include
less modules in the backward propogation process. Even if we do train on the whole text, it wouldn't change much due
to vanishing gradients.
"""

part1_q2 = r"""
As part of the training process, we keep the hidden state between batches (and apply detach() on it so it won't be
included in the BPTT). This does contribute to longer memory, without having to pay the price of backpropogating too
far back.
"""

part1_q3 = r"""
We don't shuffle because of of the importance of context. We want to make sure that each batch has the memory of the
batch before (except the first, which is negligible compared to the batch number), so we don't learn half words or 
broken phrases. We also would like to learn in a more macro level, the parts of the text and not just what character
comes after which one.
"""

part1_q4 = r"""
1.  Lowering the temperature greatly affect the variance of the output, which results in a more 'spiky' probabilities,
    for fewer characters. This ensures us more certainty of the selection of the characters. This is good for sampling 
    because we want to have text that resembles the source. While training, we prefer higher temperatures, which result
    in a more exploring nature. This variance allows us to better update the weights.

2.  When the temperature is very high we will see uniform distribution between the characters, which will result in a
    random sequence. This is because we make the differences between values smaller, which results in similar values
    when softmaxing.
    
3.  When the temperature is ver low the opposite happens. greater differences, while the softmax function is exponential
    result in very 'spiky' nature, where the smaller values get smaller, and the larger values get larger. This 
    decreases the variance and increase the probability of choosing from a small subset.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 10
    hypers['h_dim'] = 50
    hypers['z_dim'] = 15
    hypers['x_sigma2'] = 0.007
    hypers['learn_rate'] = 0.0001
    hypers['betas'] = (0.75, 0.75)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:** 
We assume that $\sigma^2$ hyperparameter is the standard deviation of $z$, the latent-space representation of $x$.
This assumption allows us to sample $z$ we sample from $\mathcal{N}( \bb{\mu} _{\bb{\alpha}}(\bb{x}),  \mathrm{diag}\{ \bb{\sigma}^2_{\bb{\alpha}}(\bb{x}) \} )$.
Choosing high value of $\sigma^2$ will result in a larger variance of the sampled $z$'s, which will lead to the larger variance of images, and vice versa. 
"""

part2_q2 = r"""
1. The first part of the loss, the reconstruction part, makes sure that the resulted images are similar the input images. 
The second part is based on the Kullback–Leibler divergence, it measures the similarity of one distribution to another.
Minimization of the second part will result by closer distributions $p(Z|X),  p(Z)$

2.  The KL loss term makes the latent space distribution similar to the assumed gaussian distribution $p(Z).

3. The sampled instances will be close to the real samples.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['z_dim'] = 100
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.3
    hypers['discriminator_optimizer'] = {'type':'SGD', 'lr':0.075}
    hypers['generator_optimizer'] = {'type':'Adam', 'lr':0.0002}
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
The gradients from the GAN are neede to train two networks.
The first - the discriminator - has its own parameters, loss and parameter's gradients.
The second - the generator - has its loss calculated according to the discriminator performence.
Thus this gradient of its parameters has to connect to it.
We implement this connection by  calculating the gradients of the generator's parameters 
so that the loss of the discriminator will be as high as possible - 
according to the negetive form of the discriminator gradients.
"""

part3_q2 = r"""
**Your answer:**
1. No, it is possible that the discriminator will improve thus, resulting with an improvement of the generator (although it might lead to a bigger loss of the generator).
1. When the discriminator loss remains at a constant value while the generator loss decreases, it is possible that the descriminator improves in identifying the real data samples while worsen in indentifying the fake samples. 
"""

part3_q3 = r"""
**Your answer:**
1. The first difference is in the quality of the generated images. The quality of the data generated from VAE is much better.
This might be a result of more suitable choosing of hyperparameters but GANs are known to be harder to train.

1. The deviation of the data generated from GAN is higher.  
Again we can say that it is the result of hyperparameters choosing, but 
the Generator did not get any image as an input and the train was solely based on the descriminator loss.
This might lead to a more "creative" model.
"""

# ==============