# Cross Modal Disentanglement:

## Overview:
	- The aim is to build a neural network that can take as input either an image or some text, and encode into a representation that has two parts - a shared part (that represents the core modality-independent information in the input) and an exclusive part (that represents the modality-specific information in the input). 
	- The shared representation can then be used for cross-modal retrieval of data; ie., given an image, finding relevant text from the database that matches the image and given some text, finding relevant images from the database that matches the text.
	- The neural network model closely follows the model used in the cross domain entanglement paper. Where necessary, the model has been adapted for the image-text scenario (whereas the paper presents the image-image scenario).

## Dataset:
	- MS-COCO Dataset with Image Caption annotations. Every image has 5 captions in the dataset.

## Networks Components:
	- ImageEncoder
		- Takes a 256x256 input image, and generates 2 vectors S_i and E_i of length $x and $y.
		- S_i represents the shared part of the representation.
		- E_i represents the exclusive part of the representation.
	- FullTextDecoder
		- Takes as input a pair of vectors S and E of lengths $x and $y respectively, concatenates them and tiles them repeatedly (vertically stacking), then uses transpose convolutions to generate text.
	- TextEncoder
		- Takes as input a sentence of max length given by $max_length, where each word is represented by a vector of length $w and generates 2 vectors S_t and E_t of length $x and $y respectively.
		- S_t is the shared part of the representation.
		- E_t is the exclusive part of the representation.
	- FullImageDecoder
		- Takes as input a pair of vectors S and E of lengths $x and $y respectively, concatenates the vectors S and E, tiles them repeatedly and uses transpose convolutions to generate an image.
	- ExclusiveTextDecoder
		- Takes as input a vector E of length $y, tiles it repeatedly (vertically stacking), then uses transpose convolutions to generate text.
	- ExclusiveImageDecoder
		- Takes as input a vector E of length $y, tiles it repeatedly and uses transpose convolutions to generate an image.
	- ExclusiveTextDiscriminator
		- Takes as input a vector of length $y and predicts whether it came from the noise distribution N(0,1) or from the exclusive text representation distribution.
	- ExclusiveImageDiscriminator
		- Takes as input a vector of length $y and predicts whether it came from the noise distribution N(0,1) or from the exclusive image representation distribution.
	- ImageDiscriminator
		- Takes as input an image I and predicts if it came from the true image distribution.
	- TextDiscriminator
		- Takes as input text T and predicts if it came from the true text distribution.

## Architecture:
	- Image2TextConverter
		- Input Image passed through ImageEncoder to get S_i and E_i.
		- E_i replaced with a randomly generated vector E_r1 of same length.
		- S_i and E_r1 passed to TextDecoder to get Text'.
		- Text' passed through TextEncoder to get S_t' and E_t'.
		- E_i passed through ExclusiveTextDecoder to generate Text''.
		- S_t' and S_i are compared for loss.
		- TextDiscriminator is applied to Text'' and Gradient reversal applied to gradients flowing from ExclusiveTextDecoder.
		- E_t' and E_r1 are passed through ExclusiveTextDiscriminator for loss.
		- TextDiscriminator is applied to Text' to compute a L_discT loss.
	- Text2ImageConverter
		- Input text passed through TextEncoder to get S_t and E_t.
		- E_t replaced with a randomly generated vector E_r2 of same length.
		- S_t and E_r2 passed through ImageDecoder to get Image'.
		- Image' passed through ImageEncoder to get S_i' and E_i'.
		- E_t passed through ExclusiveImageDecoder to generate Image''.
		- S_i' and S_t are compared for loss.
		- ImageDiscriminator is applied to Image'' and Gradient reversal applied to gradients flowing from ExclusiveImageDecoder.
		- E_i' and E_r2 are passed through ExclusiveImageDiscriminator for loss.
		- ImageDiscriminator is applied to Image' to compute a L_discI loss.
	- CrossModalImageAutoencoder and CrossModalTextAutoencoder
		- Input Image is passed through ImageEncoder to get S_i and E_i.
		- Input Text is passed through TextEncoder to get S_t and E_t.
		- S_i and E_t are passed to the FullTextDecoder to get AutoText.
		- S_t and E_i are passed to the FullImageDecoder to get AutoImage.
		- ImageDiscriminator is used to differentiate AutoImage and Image from each other.
		- TextDiscriminator is used to differentiate AutoText and Text from each other.

## Training:
	- The training will be supervised with pairs of (image, caption) as input which will be trying to optimize the following losses.
	- Losses:
		- L_s = ||S_i-S_t||
		- L_reconI = ||S_t'-S_i||
		- L_reconT = ||S_i'-S_t||
		- L_autoI = ||AutoImage-Image||
		- L_autoT = ||AutoText-Text||
		- L_genI --> tries to generate text that fools the ImageDiscriminator into thinking that the text came from the real text distribution.
		- L_discI --> tries to classify input image as belonging to image distribution or not.
		- L_ganI = L_genI + L_discI
		- L_genT --> tries to generate text that fools the TextDiscriminator into thinking that the text came from the real text distribution.
		- L_discT --> tries to classify input text as belonging to text distribution or not.
		- L_ganT = L_genT + L_discT
		- L_ganE_i --> GAN loss for Image generation from E_i.
		- L_ganE_t --> GAN loss for Text generation from E_t.
		- L_z_i --> Discriminator loss for differentiating E_i from random noise from N(0,1)
		- L_z_t --> Discriminator loss for differentiating E_t from random noise from N(0,1)
		- TotalLoss = (L_ganI + L_ganT) + (L_ganE_i + L_ganE_t) + (L_z_i + L_z_t) + (L_s + L_autoI + L_autoT + L_reconI + L_reconT)

## Links:
[Image-to-image translation for cross-domain disentanglement](http://papers.nips.cc/paper/7404-image-to-image-translation-for-cross-domain-disentanglement.pdf)

## Questions:
	- Should I get rid of L_autoI and L_autoT losses and replace them with GAN losses?
	- Should I use pre-trained models for initial features?
	- How about using Inception modules and residual connections?
	- Should I replace CNN decoder for text with GRU or LSTM? What about for the text encoder? 

## Todos:
	[] Changes to the build_mscoco_data.py script as marked by the "\#TODO:" prefix
	[] Model finalization
	[] Implementation

## Commands:
```bash
$ python build_mscoco_data.py --train_image_dir /home/ubuntu/everything/mscoco/train2014 --val_image_dir /home/ubuntu/everything/mscoco/val2014 --train_captions_file /home/ubuntu/everything/mscoco/annotations/captions_train2014.json --val_captions_file /home/ubuntu/everything/mscoco/annotations/captions_val2014.json --output_dir /home/ubuntu/everything/mscoco/output_dir
```
