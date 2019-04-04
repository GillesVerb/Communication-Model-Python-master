# Install necessary packages via: conda install --file requirements.txt

import os
from io import StringIO

# if using anaconda3 and error execute: conda install --channel conda-forge pillow=5.2.0
import numpy as np
import math
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
input_lzw = image.get_pixel_seq().copy()
# ======================= SOURCE ENCODING ========================
# ====================== Lempel-Ziv-Welch ========================
print("ingang LZW:", input_lzw)
print("lengte origineel:", len(input_lzw))
encoded_msg, dictonary = lzw.encode(input_lzw)
print("source encoded msg: ", encoded_msg)
uint8_stream = np.array(encoded_msg, dtype=np.uint8)
# ====================== CHANNEL ENCODING ========================
# ======================== Reed-Solomon ==========================
print("ingang channel encoder:", uint8_stream)
# as we are working with symbols of 8 bits
# choose n such that m is divisable by 8 when n=2^m−1
# Example: 255 + 1 = 2^m -> m = 8
n = 255  # code_word_length in symbols
k = 223  # message_length in symbols

coder = rs.RSCoder(n, k)

# generate a matrix with k rows (for each message)
uint8_stream.resize((math.ceil(len(uint8_stream)/k), k),refcheck=False)
# afterwards you can iterate over each row to encode the message
messages = uint8_stream

rs_encoded_message = StringIO()


for message in messages:
    code = coder.encode_fast(message, return_string=True)
    rs_encoded_message.write(code)

# TODO What is the RSCoder outputting? Convert to a uint8 (byte) stream before putting it over the channel
rs_encoded_message_uint8 = np.array(
    [ord(c) for c in rs_encoded_message.getvalue()], dtype=np.uint8)

print("ENCODING COMPLETE")

# TODO Use this helper function to convert a uint8 stream to a bit stream
rs_encoded_message_bit = util.uint8_to_bit(rs_encoded_message_uint8)


received_message = channel(rs_encoded_message_bit, ber=0.2) # 0.2 procent van de bits worden aangepast
                                                            # de limieten van reed solomon nog opzoeken

# TODO Use this helper function to convert a bit stream to a uint8 stream
received_message_uint8 = util.bit_to_uint8(received_message)
received_message_uint8.resize((math.ceil(len(received_message_uint8)/n), n))

decoded_message = StringIO()
print("dit is de ontvangen boodschap na RS-decodering: ", received_message_uint8)


# TODO Iterate over the received messages and compare with the original RS-encoded messages
# for cnt, (block, original_block) in enumerate(zip(received_message_uint8, rs_encoded_message_uint8)):
#     try:
#         decoded, ecc = coder.decode_fast(block, return_string=True)
#         assert coder.check(decoded + ecc), "Check not correct"
#         decoded_message.write(str(decoded))
#     except rs.RSCodecError as error:
#         diff_symbols = len(block) - (original_block == block).sum()
#         print(
#             F"Error occured after {cnt} iterations of {len(received_message_uint8)}")
#         print(F"{diff_symbols} different symbols in this block")



print("DECODING COMPLETE")


# ======================= SOURCE DECODING ========================
# ====================== Lempel-Ziv-Welch ========================

flat = received_message_uint8.flatten()
encoded_list_of_uint8 = flat.tolist()
print("flat: ", flat)
print("encoded_list_of_uint8:",encoded_list_of_uint8)
lzw_decoded_msg = lzw.decode(encoded_list_of_uint8)
print("lengte resultaat:", len(lzw_decoded_msg))
print("Resultaat:", lzw_decoded_msg)

