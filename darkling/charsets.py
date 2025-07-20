"""Predefined character sets for the darkling engine."""

# Emoji-only character set
EMOJI = (
    "😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰😗😙😚🙂🤗"
    "🤔🤨😐😑😶🙄😏😣😥😮🤐😯😪😫🥱😴😌😛😜😝"
)

# Fifteen most common special characters
COMMON_SYMBOLS = "!@#$%^&*()-=+[]"

# Uppercase and lowercase alphabets for major languages

ENGLISH_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ENGLISH_LOWER = "abcdefghijklmnopqrstuvwxyz"

SPANISH_UPPER = "AÁBCDEÉFGHIÍJKLMNÑOÓPQRSTUÚÜVWXYZ"
SPANISH_LOWER = "aábcdeéfghiíjklmnñoópqrstuúüvwxyz"

FRENCH_UPPER = "AÀÂÆBCÇDEÉÈÊËFGHIÎÏJKLMNOÔŒPQRSTUÙÛÜVWXYŸZ"
FRENCH_LOWER = "aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz"

GERMAN_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜẞ"
GERMAN_LOWER = "abcdefghijklmnopqrstuvwxyzäöüß"

ITALIAN_UPPER = "AÀBCDEÉÈFGHIÌÍJKLMNOÒÓPQRSTUÙUVWXYZ"
ITALIAN_LOWER = "aàbcdeéèfghiìíjklmnoòópqrstuùuvwxyz"

PORTUGUESE_UPPER = "AÁÂÃÀBCÇDEÉÊFGHIÍJKLMNOÓÔÕPQRSTUÚÜVWXYZ"
PORTUGUESE_LOWER = "aáâãàbcçdeéêfghiíjklmnoóôõpqrstuúüvwxyz"

DUTCH_UPPER = "AÁBCDEÉFGHIÏJKLMNOÓPQRSTUÜVWXYZ"
DUTCH_LOWER = "aábcdeéfghiïjklmnoópqrstuüvwxyz"

SWEDISH_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ"
SWEDISH_LOWER = "abcdefghijklmnopqrstuvwxyzåäö"

NORWEGIAN_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ"
NORWEGIAN_LOWER = "abcdefghijklmnopqrstuvwxyzæøå"

DANISH_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ"
DANISH_LOWER = "abcdefghijklmnopqrstuvwxyzæøå"

FINNISH_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ"
FINNISH_LOWER = "abcdefghijklmnopqrstuvwxyzåäö"

POLISH_UPPER = "AĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŹŻ"
POLISH_LOWER = "aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż"

CZECH_UPPER = "AÁBCČDĎEÉĚFGHIÍJKLMNŇOÓPQRŘSŠTŤUÚŮVWXYZÝŽ"
CZECH_LOWER = "aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyzýž"

HUNGARIAN_UPPER = "AÁBCDEÉFGHIÍJKLMNOÓÖŐPQRSTUÚÜŰVWXYZ"
HUNGARIAN_LOWER = "aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz"

ROMANIAN_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZĂÂÎȘȚ"
ROMANIAN_LOWER = "abcdefghijklmnopqrstuvwxyzăâîșț"

TURKISH_UPPER = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
TURKISH_LOWER = "abcçdefgğhıijklmnoöprsştuüvyz"

GREEK_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
GREEK_LOWER = "αβγδεζηθικλμνξοπρστυφχψω"

RUSSIAN_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
RUSSIAN_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

# Languages without case
ARABIC = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
HINDI = "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
CHINESE = (
    "的一是不了在人这有他我中大来上国到说们为子和你地出道也时年得就那要下以生"
    "会自着去之过家学对可看她里后小么心多天"
)
JAPANESE_HIRAGANA = (
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ"
    "まみむめもやゆよらりるれろわをん"
)
JAPANESE_KATAKANA = (
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホ"
    "マミムメモヤユヨラリルレロワヲン"
)
JAPANESE = JAPANESE_HIRAGANA + JAPANESE_KATAKANA
