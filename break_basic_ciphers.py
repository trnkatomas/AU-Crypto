import re
import matplotlib.pyplot as plt
import random

"""Affine Cipher"""

cryptoText = """KQEREJEBCPPCJCRKIEACUZBKRVPKRBCIBQCARBJCVFCUP
KRIOFKPACUZQEPBKRXPEIIEABDKPBCPFCDCCAFIEABDKP
BCPFEQPKAZBKRHAIBKAPCCIBURCCDKDCCJCIDFUIXPAFF
ERBICZDFKABICBBENEFCUPJCVKABPCYDCCDPKBCOCPERK
IVKSCPICBRKIJPKABI"""

unknownCipher = """BNVSNSIHQCEELSSKKYERIFJKXUMBGYKAMQLJTYAVFBKVT
DVBPVVRJYYLAOKYMPQSCGDLFSRLLPROYGESEBUUALRWXM
MASAZLGLEDFJBZAVVPXWICGJXASCBYEHOSNMULKCEAHTQ
OKMFLEBKFXLRRFDTZXCIWBJSICBGAWDVYDHAVFJXZIBKC
GJIWEAHTTOEWTUHKRQVVRGZBXYIREMMASCSPBNLHJMBLR
FFJELHWEYLWISTFVVYFJCMHYUYRUFSFMGESIGRLWALSWM
NUHSIMYYITCCQPZSICEHBCCMZFEGVJYOCDEMMPGHVAAUM
ELCMOEHVLTIPSUYILVGFLMVWDVYDBTHFRAYISYSGKVSUU
HYHGGCKTMBLRX"""

vigenereCipher = """KCCPKBGUFDPHQTYAVINRRTMVGRKDNBVFDETDGILTXRGUD
DKOTFMBPVGEGLTGCKQRACQCWDNAWCRXIZAKFTLEWRPTYC
QKYVXCHKFTPONCQQRHJVAJUWETMCMSPKQDYHJVDAHCTRL
SVSKCGCZQQDZXGSFRLSWCWSJTBHAFSIASPRJAHKJRJUMV
GKMITZHFPDISPZLVLGWTFPLKKEBDPGCEBSHCTJRWXBAFS
PEZQNRWXCVYCGAONWDDKACKAWBBIKFTIOVKCGGHJVLNHI
FFSQESVYCLACNVRWBBIREPBBVFEXOSCDYGZWPFDTKFQIY
CWHJVLNHIQIBTKHJVNPIST"""

vige_ex = """CHREE VOAHM AERAT BIAXX WTNXB EEOPH BSBQMQEQERBW
RVXUO AKXAO SXXWE AHBWG JMMQM NKGRFV GXWTRZXWIAK
LXFPS KAUTE MNDCM GTSXM XBTUI ADNGM GPSRELXNJELX
VRVPR TULHD NQWTW DTYGB PHXTF ALJHA SVBFXNGLLCHR
ZBWEL EKMSJ IKNBH WRJGN MGJSG LXFEY PHAGNRBIEQJT
AMRVL CRREM NDGLX RRIMG NSNRW CHRQH AEYEVTAQEBBI
PEEWE VKAKO EWADR EMXMT BHHCH RTKDN VRZCHRCLQOHP
WQAII WXNRM GWOII FKEE"""

streamCipher = """IYMYSILONRFNCQXQJEDSHBUIBCJUZBOLFQYSCHATPEQGQ
JEJNGNXZWHHGWFSUKULJQACZKKJOAAHGKEMTAFGMKVRDO
PXNEHEKZNKFSKIFRQVHHOVXINPHMRTJPYWQGJWPUUVKFP
OAWPMRKKQZWLQDYAZDRMLPBJKJOBWIWPSEPVVQMBCRYVC
RUZAAOUMBCHDAGDIEMSZFZHALIGKEMJJFPCIWKRMLMPIN
AYOFIREAOLDTHITDVRMSE"""

cryptoText = re.sub(r"\s+", '', cryptoText)
vigenereCipher = re.sub(r"\s+", '', vigenereCipher)
vige_ex = re.sub(r"\s+", '', vige_ex)
unknownCipher = re.sub(r"\s+", '', unknownCipher)
streamCipher = re.sub(r"\s+", '', streamCipher)

letters_prob = ['E', 'T']


def show_histogram(text, title=None):
    plt.figure()
    u_dict = freq_analysis(text)
    s_dict = sort_dict_by_value(u_dict)
    rects = plt.bar(range(len(s_dict)), [x[1] for x in s_dict], align='center')
    autolabel(rects)
    plt.xticks(range(len(s_dict)), [x[0] for x in s_dict])
    py_num = random.randint(1, 1000)
    fr_h = plt.plot()
    if title:
        plt.title(title)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 0.1 + height, '%d' % int(height),
                 ha='center', va='bottom')


def freq_analysis(text):
    freq_dict = {}
    for i in range(65, 91):
        freq_dict[chr(i)] = sum([1 if ord(letter) == i else 0 for letter in text])
    return freq_dict


def sort_dict_by_value(dictionary):
    return sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

def stream_cipher_enc(key,Z,text):
    key_length = len(key)
    key = chars_to_numbers(key)
    num_text = chars_to_numbers(text)
    encryp_text = [(num_text[i] + key[i%key_length]+1*(i//key_length)%26) % Z for i in range(len(num_text))]
    return "".join(numbers_to_chars(encryp_text))

def stream_cipher_dec(key,Z,text):
    key_length = len(key)
    key = chars_to_numbers(key)
    num_text = chars_to_numbers(text)
    encryp_text = [(num_text[i] - key[i%key_length]-1*(i//key_length)%26) % Z for i in range(len(num_text))]
    return "".join(numbers_to_chars(encryp_text))

def vigenere_cipher_enc(key,Z,text):
    key_length = len(key)
    key = chars_to_numbers(key)
    num_text = chars_to_numbers(text)
    encryp_text = [(num_text[i] + key[i%key_length]) % Z for i in range(len(num_text))]
    return "".join(numbers_to_chars(encryp_text))

def vigenere_cipher_dec(key,Z,text):
    key_length = len(key)
    key = chars_to_numbers(key)
    num_text = chars_to_numbers(text)
    encryp_text = [(num_text[i] - key[i%key_length]) % Z for i in range(len(num_text))]
    return "".join(numbers_to_chars(encryp_text))

def affine_cipher_enc(a, b, Z, text):
    encryp_text = [(a * letter + b) % Z for letter in chars_to_numbers(text)]
    return "".join(numbers_to_chars(encryp_text))

def affine_cipher_dec(a, b, Z, text):
    a_inv = modinv(a, Z)
    decryp_text = [a_inv * (letter - b) % Z for letter in chars_to_numbers(text)]
    return "".join(numbers_to_chars(decryp_text))

def shift_cipher_enc(a, Z, text):
    encryp_text = [(a + letter) % Z for letter in chars_to_numbers(text)]
    return "".join(numbers_to_chars(encryp_text))

def shift_cipher_dec(a, Z, text):
    decryp_text = [ (letter - a) % Z for letter in chars_to_numbers(text)]
    return "".join(numbers_to_chars(decryp_text))

def break_affine_cipher(Z, text):
    f_dict = freq_analysis(text)
    sorted_f_dict = sort_dict_by_value(f_dict)
    letters_prob = ['E', 'T']
    for (a, b, e) in compute_shift(sorted_f_dict, letters_prob, Z, text):
        if e:
            #print(e)
            pass
        else:
            a_inv = modinv(a, Z)
            #print(text)
            print(affine_cipher_dec(a, b, Z, text))
    """(a,b) = compute_shift(sorted_f_dict,letters_prob,Z,text)
	print(a,b)
	a_inv = modinv(a,Z)
	print(affine_cipher_dec(a,b,Z,text))
	return affine_cipher_dec(a,b,Z,text)"""


def numbers_to_chars(text):
    return [chr(number + 65) for number in text]


def chars_to_numbers(text):
    return [ord(letter) - 65 for letter in text]


def solve_eqation(x_1, x_2, y_1, y_2, Z):
    x = (x_1 - x_2) % Z
    y = (y_1 - y_2) % Z
    x_inv = modinv(x, Z)
    a = x_inv * y % Z
    b = (y_1 - (x_1 * a)) % Z
    return (a, b)

def compute_shift(ord_dict, letter_freq, Z, text):
    for i in range(5):  # range(len(ord_dict)):
        x_1 = ord(letter_freq[0])
        y_1 = ord_dict[i][1] - x_1
        """print("assigned", letter_freq[0], " to ", ord_dict[i][0])
        print("the shift is: ", y_1 % 26)
        print("a*", ord(letter_freq[0]) - 65, "+b = ", ord(ord_dict[i][0]) - 65)     """
        for j in range(i + 1, 5):  #range(i+1,len(ord_dict)):
            x_2 = ord(letter_freq[1])
            y_2 = ord_dict[j][1] - x_2
            """print("assigned", letter_freq[1], " to ", ord_dict[j][0])
            print("the shift is: ", y_2 % 26)
            print("a*", ord(letter_freq[1]) - 65, "+b = ", ord(ord_dict[j][0]) - 65)"""
            (a, b) = solve_eqation(x_1, x_2, y_1, y_2, Z)
            g, _x, _y = bezout(a, Z)
            if g != 1:
                yield (None, None, """An "a" has no inverse in Z""")
            else:
                #print("a:", a, " b: ", b)
                yield (a, b, None)


def bezout(aa, bb):
    r, r_0 = abs(aa), abs(bb)
    x_0, x, y_0, y = 0, 1, 1, 0
    while r_0:
        r, (q, r_0) = r_0, divmod(r, r_0)
        x_0, x = x - q * x_0, x_0
        y_0, y = y - q * y_0, y_0
        # print("x_0: ",x_0," x: ",x," y_0: ",y_0," y: ",y, " r_0: ",r_0, " r: ",r," q: ",q)
    return r, x * (-1 if aa < 0 else 1), y * (-1 if bb < 0 else 1)


def modinv(a, m):
    g, x, y = bezout(a, m)
    if g != 1:
        raise ValueError
    return x % m


def write_in_blocks(text, cols):
    counter = 0
    for letter in text:
        print("{}".format(letter), end="")
        counter += 1
        if counter % cols == 0:
            print()
    print()


def compute_column_probability(text, cols):
    cols_T = [[] for x in range(cols)]
    counter = 0
    for letter in text:
        cols_T[counter].append(letter)
        counter += 1
        counter %= cols
    sum = 0
    for col in cols_T:
        text = "".join(col)
        val = compute_Ic(text)
        sum += val
        #print("{:.3f}".format(val), end=" ")
    #print()
    print("Average value: ", sum/len(cols_T))
    return (sum/len(cols_T),cols_T)


def compute_Ic(text):
    f_dict = freq_analysis(text)
    suma = 0
    for i in range(65, 65 + 26 + 1):
        if chr(i) in f_dict:
            suma += f_dict[chr(i)] * (f_dict[chr(i)] - 1)
        # print(suma)
    return suma / (len(text) * (len(text) - 1))

def compute_Ic_shifted(text,shift):
    f_dict = freq_analysis(text)
    probability_table = {'A':0.082,'B':0.015,'C':0.028,'D':0.043,'E':0.127,
                         'F':0.022,'G':0.020,'H':0.061,'I':0.070,'J':0.002,
                         'K':0.008,'L':0.040,'M':0.024,'N':0.067,'O':0.075,
                         'P':0.019,'Q':0.001,'R':0.060,'S':0.063,'T':0.091,
                         'U':0.028,'V':0.010,'W':0.023,'X':0.001,'Y':0.020,'Z':0.001}
    suma = 0
    for i in range(26):
        if chr((i+shift)%26+65) in f_dict:
            suma += probability_table[chr(i+65)]*f_dict[chr((i+shift)%26+65)]
    return suma / len(text)

def stream_shift(text,length):
    new_crypto = [0 for x in range(len(text))]
    j = 1
    for l in range(len(text)):
        new_crypto[l] = (ord(text[l])-65 - j*(l//length))%26
    new_crypto = "".join(numbers_to_chars(new_crypto))
    print("new crypto: ",new_crypto)
    return new_crypto

def try_stream(crypto_text):
    cols_array = []
    for i in range(1, 11):
        #print("Column width: ",i)
        print(str(i) + " ", end="")
        new_crypto = stream_shift(crypto_text,i)
        (val,cols) = compute_column_probability(new_crypto, i)
        cols_array.append((i,val,cols))

    cols_array = sorted(cols_array, key=lambda x: x[1], reverse=True)

    print("Best block size is: ", cols_array[0][0])

    for (i,val, cols) in cols_array:
        cols_best = []
        for col in cols:
            text = "".join(col)
            occ = []
            for j in range(26):
                occ.append(compute_Ic_shifted(text,j))
            cols_best.append(occ.index(max(occ)))
        best_guess_key = "".join(numbers_to_chars(cols_best))
        print(cols_best)
        print(best_guess_key)
        print(stream_cipher_dec(best_guess_key,26,crypto_text))


def try_vigenere(crypto_text):
    cols_array = []
    for i in range(1, 11):
        #print("Column width: ",i)
        print(str(i) + " ", end="")
        (val,cols) = compute_column_probability(crypto_text, i)
        cols_array.append((i,val,cols))

    cols_array = sorted(cols_array, key=lambda x: x[1], reverse=True)

    print("Best block size is: ", cols_array[0][0])

    for (i,val, cols) in cols_array:
        cols_best = []
        for col in cols:
            text = "".join(col)
            occ = []
            for j in range(26):
                occ.append(compute_Ic_shifted(text,j))
            cols_best.append(occ.index(max(occ)))
        best_guess_key = "".join(numbers_to_chars(cols_best))
        print(cols_best)
        print(best_guess_key)
        print(vigenere_cipher_dec(best_guess_key,26,crypto_text))


f_dict = freq_analysis(cryptoText)
print(f_dict)
print(sort_dict_by_value(f_dict))
print(affine_cipher_dec(3, 5, 26, affine_cipher_enc(3, 5, 26, "ALGORITHMSARE")))
print(solve_eqation(4, 19, 17, 3, 26))
# break_affine_cipher(26,cryptoText)
#print("""O CANADA TERRE DE NOS AIEUX T ON FRONT EST CE INT DE FLEUR ONS GLORIEUX CAR T ON BRASSAIT PORTER LE PEEILS AI T PORTER LA CROIX T ON HISTOIRE EST UNE EPOPEE DES PLUS BRILLANTS EXPLOIT SET TA VALEUR DE FOIT REMPEE PROTEGERA NOS FOYERS ET NOS DROITS""")

#show_histogram(cryptoText,title='Affine cipher')
show_histogram(unknownCipher, title='unknown')
#show_histogram(vigenereCipher,title='vigenere')
#plt.show()
plt.savefig('freq_u.png')
print("\n\n#####################################\n\n")
#print(compute_Ic(cryptoText))
[print(x, end=";") for x in range(11)]
print()

#try_vigenere(unknownCipher)
#try_stream(streamCipher)
#print(vigenere_cipher_dec("CIPHER",26,vigenere_cipher_enc("CIPHER",26,"THISCRYPTOSYSTEMISNOTSECURE")))
#print(vigenere_cipher_dec("JANET",26,vige_ex))
#print(shift_cipher_dec(11,26,shift_cipher_enc(11,26,"WEWILLMEETATMIDNIGHT")))

probability_table = {'A':0.082,'B':0.015,'C':0.028,'D':0.043,'E':0.127,
                     'F':0.022,'G':0.020,'H':0.061,'I':0.070,'J':0.002,
                     'K':0.008,'L':0.040,'M':0.024,'N':0.067,'O':0.075,
                     'P':0.019,'Q':0.001,'R':0.060,'S':0.063,'T':0.091,
                     'U':0.028,'V':0.010,'W':0.023,'X':0.001,'Y':0.020,'Z':0.001}
import math

H = 0
for i in probability_table.keys():
    h = -probability_table[i]*math.log(probability_table[i],2)
    print(i,":",h)
    H += h
    #h = -(1-probability_table[i])*math.log(1-probability_table[i],2)
    print("non",i,":",h)
    #H += h#-(1-probability_table[i])*math.log(1-probability_table[i],2)
print("Entropy H:", H)
#print("Avg:",H/len(probability_table.keys()))