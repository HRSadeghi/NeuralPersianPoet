#Copyright 2022 Hamidreza Sadeghi. All rights reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


import string , re
from unidecode import unidecode
from tqdm import tqdm

def per_num_to_eng(string):
  out = ""
  num_dic = {
            '۰': '0',
            '۱': '1',
            '۲': '2',
            '۳': '3',
            '۴': '4',
            '۵': '5',
            '۶': '6',
            '۷': '7',
            '۸': '8',
            '۹': '9',
        }
  for x in string:
    if x in num_dic.keys():
      out += num_dic[x]
    else:
      out += x
  return out


def replace(x):
    if x == 'ك':
        return 'ک'
    elif x == 'ي':
        return 'ی'
    elif x == 'إ':
        return 'ا'
    elif x == 'ۀ':
        return ' '
    elif x == 'أ':
        return 'ا'
    elif x == 'ٱ':
        return 'ا'
    elif x == 'ٲ':
        return 'ا'
    elif x == 'ړ':
        return 'ر'
    elif x == 'ۀ':
        return 'ه'
    elif x == 'څ':
        return 'خ'
    elif x == 'ہ':
        return 'ه'
    elif x == 'ګ':
        return 'گ'
    elif x == 'ة':
        return 'ت'
    elif x == 'ى':
        return 'ی'
    elif x == 'ٖ':
        return ' '
    elif x ==' ٕٕ':
        return ' '
    elif x == 'ڱ':
        return 'گ'
    elif x == 'ڪ':
        return 'ک'
    elif x == 'ھ':
        return 'ه'
    elif x == 'ۂ':
        return 'ه'
    elif x == 'ۆ':
        return 'و'
    elif x == 'ہ':
        return 'ه'
    elif x == 'ۀ':
        return 'ه'
    elif x == 'ڄ':
        return 'ج'
    elif x == 'ٹ':
        return 'ت'
    elif x == '٭':
        return ' '
    elif x == 'ٴ':
        return ''
    elif x == ' ٕ':
        return ' '
    elif x == 'ٶ':
        return 'ؤ'
    elif x == 'ڤ':
        return 'ف'
    elif x == 'ٕ':
        return ''
    elif x =='ء':
      return ' ی'
    elif x == '٪':
      return ' درصد'
    elif x == '%':
      return ' درصد'
    elif x == '$':
      return ' دلار'
    elif x == 'أ':
      return 'ا'
    elif x == 'ئ':
      return 'ی'
    elif x == 'ؤ':
      return 'و'
    elif x == 'ئ':
      return 'ی'
    elif x == 'ﮐ':
      return 'ک'
    elif x == 'ﮔ':
      return 'گ'
    elif x == 'ﯾ':
      return 'ی'
    elif x == 'ﯿ':
      return 'ی'
    elif x == 'ﻤ':
      return 'م'
    elif x == 'ﻮ':
      return 'و'
    elif x == 'ﻭ':
      return 'و'
    elif x == 'ﻧ':
      return 'ن'
    elif x == 'ﻌ':
      return 'ع'
    elif x == 'ﻋ':
      return 'ع'
    elif x == 'ﺸ':
      return 'ش'
    elif x == 'ﺷ':
      return 'ش'
    elif x == 'ﺴ':
      return 'س'
    elif x == 'ﺮ':
      return 'ر'
    elif x == 'ﺭ':
      return 'ر'
    elif x == 'ﺪ':
      return 'د'
    elif x == 'ﺩ':
      return 'د'
    elif x == 'ﺧ':
      return 'خ'
    elif x == 'ﺘ':
      return 'ت'
    elif x == 'ﺖ':
      return 'ت'
    elif x == 'ﺒ':
      return 'ب'
    elif x == 'ﺑ':
      return 'ب'
    elif x == 'ﺎ':
      return 'ا'
    elif x == 'ﺍ':
      return 'ا'
    elif x == 'ﯿ':
      return 'ی'
    elif x == 'ﯾ':
      return 'ی'
    elif x == 'ﻢ':
      return 'م'
    elif x == 'ﻥ':
      return 'ن'
    elif x == 'ّ':
      return ""
    elif x == "ِ":
      return ""
    elif x == "َ":
      return ""
    elif x == "ُ":
      return ""
    elif x == "ٔ":
      return ""
    elif x == "َ":
      return ""
    elif x == "ٌ":
      return ""
    elif x == "ً":
      return ""
    elif x == '–':
      return " "
    elif x == 'ڈ':
        return "د"
    elif x == 'ڵ':
        return "ل"
    elif x == 'ٻ':
        return "ب"
    elif x == 'پ':
        return "پ"
    elif x == 'ځ':
        return "خ"
    elif x == 'چ':
        return "چ"
    elif x == 'ں':
        return "ن"
    elif x == '\u200c':
        return " "
    elif x == '\xad':
        return ''
    elif x ==  'ٔ':
        return ""
    else:
        return x

def clean(text):
  try:
    text = str(text)
    # Convert text to lowercase
    text = text.lower()

    text = re.sub("'", '', text)
    # Process commas
    text = re.sub("'", '', text)
    # Get rid of punctuation
    text=''.join(ch for ch in text if ch not in set(string.punctuation))
    #text=''.join([i for i in text if not i.isdigit()])
    text =''.join([ unidecode(x) if x.isdigit() else x for x in text])
    #persian_alpha_codepoints = '\u0621-\u0628\u062A-\u063A\u0641-\u0642\u0644-\u0648\u064E-\u0651\u0655\u067E\u0686\u0698\u06A9\u06AF\u06BE\u06CC'
    #text= re.sub('[^'+persian_alpha_codepoints+']'," ", text)
    text = ''.join([replace(x) for x in text])
    #text = " ".join(re.split("[^آ-ی0-9]*", text)) 
    #text = " ".join(text.split())
    # Get rid of Persian punctuation
    
    # Don't touch this line unless you understand how unicode works!
    for char in "[-=!؟٬٫﷼٪×،*\)\(ـ+ًٌٍَُِّْ]}{؛:«ٰ»ٰٓ‌ٔء><؟؟}٪]":
      text = text.replace(char, '')
  
    text = text.strip()
    #text = re.sub("[٠١٢٣٤٥٦٧٨٩]", '', text)
    #text = re.sub("[۱۲۳۴۵۶۷۸۹۰]", '', text)
           
    return text
  except Exception as e:
    print(e.message)
    return text


def remove_extra_tokens(input_, 
                        extra_tokens = ["__sow", "__eow", "__pad", "__she", '__unk', '__ehe1', '__ehe2', '__epm', '__spm']):
    out = ' '.join([x for x in input_.split(' ') if x not in extra_tokens])
    for x in extra_tokens:
        out = out.replace(x, ' ')
    return out