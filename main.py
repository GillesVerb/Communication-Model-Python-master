# Install necessary packages via: conda install --file requirements.txt

import os
from io import StringIO

# if using anaconda3 and error execute: conda install --channel conda-forge pillow=5.2.0
import numpy as np
import math
import huffman
import lzw
import util
from channel import channel
from imageSource import ImageSource
from unireedsolomon import rs
from util import Time

# ========================= SOURCE =========================
# Select an image - Done
IMG_NAME = 'stack2.jpg'

dir_path = os.path.dirname(os.path.realpath(__file__))
IMG_PATH = os.path.join(dir_path, IMG_NAME)  # use absolute path

print(F"Loading {IMG_NAME} at {IMG_PATH}")
image = ImageSource().load_from_file(IMG_PATH)
print(image)
# uncomment if you want to display the loaded image
# image.show()
# uncomment if you want to show the histogram of the colors
# image.show_color_hist()

# ================================================================

# ======================= SOURCE ENCODING ========================
# =========================== Huffman ============================

# Use t.tic() and t.toc() to measure the executing time as shown below

t = Time()
t.tic()
# Determine the number of occurrences of the source or use a fixed huffman_freq
uniekeWaardes, aantalOccurs = np.unique(image.get_pixel_seq(), return_counts=True)
freq=list(zip(uniekeWaardes,aantalOccurs))
huffman_freq = freq # Op basis van histogram van foto stack.jpg
huffman_tree = huffman.Tree(huffman_freq)
print(F"Generating the Huffman Tree took {t.toc_str()}")

t.tic()
#  print-out the codebook and validate the codebook (include your findings in the report)
encoded_message = huffman.encode(huffman_tree.codebook, image.get_pixel_seq())
print(F"dit is de codebook (hoop ik): {huffman_tree.codebook}")
print(len(encoded_message))
print(F"Enc: {t.toc_str()}")
print(huffman_tree.print())


print("Dit is de Huffman geëncodeerde data:", encoded_message)
t.tic()
decoded_message = huffman.decode(huffman_tree, encoded_message)
print(F"Dec: {t.toc_str()}")
print("len image:", len(image.get_pixel_seq()))
print("len dec msg huff", len(decoded_message))

# ======================= SOURCE ENCODING ========================
# ====================== Lempel-Ziv-Welch ========================
input_lzw = image.get_pixel_seq().copy()
t.tic()
encoded_msg, dictonary = lzw.encode(input_lzw)
print("Enc: {}".format(t.toc()))
print("dit is het geëncodeerde LZW formaat: ", encoded_msg)
t.tic()
decoded_msg = lzw.decode(encoded_msg)
print("Enc: {0:.4f}".format(t.toc()))
print("formaat decoded LZW:",decoded_msg)
print("len dec msg LZW", len(decoded_message))
uint8_stream = np.array(decoded_msg, dtype=np.uint8)
# ====================== CHANNEL ENCODING ========================
# ======================== Reed-Solomon ==========================
print("dit formaat komt binnen in het channel:", uint8_stream)
print("en dit is de lengte van deze data", len(uint8_stream))
# as we are working with symbols of 8 bits
# choose n such that m is divisible by 8 when n=2^m−1
# Example: 255 + 1 = 2^m -> m = 8
n = 255  # code_word_length in symbols
k = 223  # message_length in symbols

coder = rs.RSCoder(n, k)

# generate a matrix with k rows (for each message)
uint8_stream.resize((math.ceil(len(uint8_stream)/k), k), refcheck=False)
# afterwards you can iterate over each row to encode the message
messages = uint8_stream

rs_encoded_message = StringIO()

t.tic()
for message in messages:
    code = coder.encode_fast(message, return_string=True)
    rs_encoded_message.write(code)

# TODO What is the RSCoder outputting? Convert to a uint8 (byte) stream before putting it over the channel
rs_encoded_message_uint8 = np.array(
    [ord(c) for c in rs_encoded_message.getvalue()], dtype=np.uint8)
print(t.toc())
print("ENCODING COMPLETE")

# TODO Use this helper function to convert a uint8 stream to a bit stream7
print("dit bericht wordt omgezet van uint8 naar bitstream: ", rs_encoded_message_uint8)
rs_encoded_message_bit = util.uint8_to_bit(rs_encoded_message_uint8)
print("bitstream voor ber:", rs_encoded_message_bit)

t.tic()
received_message = channel(rs_encoded_message_bit, ber=0.05)# 0.5 procent van de bits worden aangepast
t.toc_print()                                            # de limieten van reed solomon nog opzoeken

# TODO Use this helper function to convert a bit stream to a uint8 stream
received_message_uint8 = util.bit_to_uint8(received_message)
received_message_uint8.resize((math.ceil(len(received_message_uint8)/n),n))
decoded_message = StringIO()
t.tic()

# TODO Iterate over the received messages and compare with the original RS-encoded messages
for cnt, (block, original_block) in enumerate(zip(received_message_uint8, rs_encoded_message_uint8)):
    try:
        decoded, ecc = coder.decode_fast(block, return_string=True)
        assert coder.check(decoded + ecc), "Check not correct"
        decoded_message.write(str(decoded))
    except rs.RSCodecError as error:
        diff_symbols = len(block) - (original_block == block).sum()
        print(
            F"Error occured after {cnt} iterations of {len(received_message_uint8)}")
        print(F"{diff_symbols} different symbols in this block")

t.toc_print()

print("DECODING COMPLETE")
print("dit is de lengte van de data die uit de kanaalcodering komt:", len(received_message))

# TODO after everything works, try to simulate the communication model as specified in the assingment
