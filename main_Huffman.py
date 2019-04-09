# Install necessary packages via: conda install --file requirements.txt

import os
from io import StringIO
from PIL import Image

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
#image.show()
# uncomment if you want to show the histogram of the colors
#image.show_color_hist()

# ================================================================

# ======================= SOURCE ENCODING ========================
# =========================== Huffman ============================
print("-------START HUFFMAN ENCODING-------")
# Use t.tic() and t.toc() to measure the executing time as shown below
print("lengte initiele data:", (8 * len(image.get_pixel_seq())), " bits")
t = Time()
t.tic()
# Determine the number of occurrences of the source or use a fixed huffman_freq
uniekeWaardes, aantalOccurs = np.unique(image.get_pixel_seq(), return_counts=True)
freq = list(zip(uniekeWaardes, aantalOccurs))
huffman_freq = freq # TO DO op basis van histogram van foto stack.jpg
huffman_tree = huffman.Tree(huffman_freq)
print(F"Generating the Huffman Tree took {t.toc_str()}")

t.tic()
#  print-out the codebook and validate the codebook (include your findings in the report)
encoded_message = huffman.encode(huffman_tree.codebook, image.get_pixel_seq())
# print(F"dit is de codebook (hoop ik): {huffman_tree.codebook}")
print("lengte encoded msg:", len(encoded_message))
print(F"Enc: {t.toc_str()}")
# print(huffman_tree.print())

t.tic()
#print("encoded msg is stream van enen en nullen")

bitstream_naar_intstream = util.bit_to_uint8(encoded_message)

uint8_stream = bitstream_naar_intstream
#print("dit gaat binnen in channel encoding:", uint8_stream)
# ====================== CHANNEL ENCODING ========================
# ======================== Reed-Solomon ==========================
print("-------START CHANNEL ENCODING-------")
initiele_lengte = len(uint8_stream)
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


for message in messages:
    code = coder.encode_fast(message, return_string=True)
    rs_encoded_message.write(code)

# TODO What is the RSCoder outputting? Convert to a uint8 (byte) stream before putting it over the channel
rs_encoded_message_uint8 = np.array(
    [ord(c) for c in rs_encoded_message.getvalue()], dtype=np.uint8)

print("CHANNEL ENCODING COMPLETE")

# TODO Use this helper function to convert a uint8 stream to a bit stream
rs_encoded_message_bit = util.uint8_to_bit(rs_encoded_message_uint8)
print("-------START CHANNEL-------")

received_message = channel(rs_encoded_message_bit, ber=0.1)#0.1 procent van de bits worden aangepast
                                                           # de limieten van reed solomon nog opzoeken
print("De lengte van de data die op het kanaal komt:", len(received_message), "bits")
# TODO Use this helper function to convert a bit stream to a uint8 stream
received_message_uint8 = util.bit_to_uint8(received_message)

received_message_uint8.resize((math.ceil(len(received_message_uint8)/n), n))

decoded_message = StringIO()
print("Het aantal toegevoegde bits is:",(len(received_message)-(8*initiele_lengte)))
print("-------START CHANNEL DECODING-------")
t.tic()
# TODO Iterate over the received messages and compare with the original RS-encoded messages
for cnt, (block, original_block) in enumerate(zip(received_message_uint8, rs_encoded_message_uint8)):
    try:
        decoded, ecc = coder.decode_fast(block, True, return_string=True)
        assert coder.check(decoded + ecc), "Check not correct"
        decoded_message.write(str(decoded))
    except rs.RSCodecError as error:
        diff_symbols = len(block) - (original_block == block).sum()
        print(
            F"Error occured after {cnt} iterations of {len(received_message_uint8)}")
        print(F"{diff_symbols} different symbols in this block")


decoded_message_uint8 = np.array(
    [ord(c) for c in decoded_message.getvalue()], dtype=np.uint8)

# Overbodige data wissen
te_vertwijderen_nullen = len(decoded_message_uint8) - initiele_lengte
print("Er moeten ", te_vertwijderen_nullen,"bits verwijderd worden")

decoded_message_uint8 = decoded_message_uint8[:-te_vertwijderen_nullen or None]
#print("decoded_message_uint8: ", decoded_message_uint8)
#print("Lengte hiervan is:", (8*len(decoded_message_uint8)),"bits")
print("Het verschil voor en na kanaaldecodering en verwijderde bits is: ", len(decoded_message_uint8) - initiele_lengte)


# ======================= SOURCE DECODING ========================
# =========================== Huffman ============================
print("-------START HUFFMAN DECODING-------")

print("lengte chan decoded data", (8*len(decoded_message_uint8)),"bits")
klaar_voor_src_dec = util.uint8_to_bit(decoded_message_uint8)
huf_decoded_message = huffman.decode(huffman_tree, klaar_voor_src_dec)
print(F"Dec: {t.toc_str()}")
print("Huffman decoded lengte:", (8*len(huf_decoded_message)), "bits, = lengte originele data")

# ======================= Source recreating ========================
print("-------START SOURCE RECREATING-------")
verhouding = np.reshape(huf_decoded_message, (image.height, image.width, image.num_of_channels))

afbeelding = Image.fromarray(verhouding,image.mode)
afbeelding.show()