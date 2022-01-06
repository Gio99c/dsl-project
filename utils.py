


CONTRACTIONS = { 
    "ain't": "are not",
    "aint": "are not",
    "aren't": "are not",
    "arent": "are not",
    "can't": "can not",
    "cant": "can not",
    "can't've": "can not have",
    "cant've": "can not have",
    "'cause": "because",
    "cause": "because",
    "could've": "could have",
    "couldve": "could have",
    "couldn't": "could not",
    "couldnt": "could not",
    "couldn't've": "could not have",
    "couldnt've": "could not have",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont": "do not",
    "hadn't": "had not",
    "hadnt": "had not",
    "hadn't've": "had not have",
    "hadnt've": "had not have",
    "hasn't": "has not",
    "hasnt": "has not",
    "haven't": "have not",
    "havent": "have not",
    "he'd": "he would",
    "hed": "he would",
    "he'd've": "he would have",
    "hed've": "he would have",
    "he'll": " he will",
    "he'll've": " he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "hows": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "isnt": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "lets": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "maynt": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustnt": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "neednt": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": " she will",
    "she'll've": " she will have",
    "she's": " she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": " so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": " there is",
    "they'd": " they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": " they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": " what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": " who will",
    "who'll've": " who will have",
    "who's": " who is",
    "who've": "who have",
    "why's": " why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": " you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

STOPWORDS = [
    'what',
    'which',
    'who',
    'whom',
    'this',
    'that',
    "that'll",
    'these',
    'those',
    'am',
    'is',
    'are',
    'was',
    'were',
    'be',
    'been',
    'being',
    'have',
    'has',
    'had',
    'having',
    'does',
    'did',
    'doing',
    'a',
    'an',
    'the',
    'and',
    'or',
    'as',
    'of',
    'at',
    'by',
    'for',
    'to',
    'from',
    'then',
    'there',
    'how',
    's',
    't',
    ]

SLANGS = {'2day': 'today',
 '2nite': 'tonight',
 '4u': 'for you',
 '4ward': 'forward',
 'a3': 'anyplace, anywhere, anytime',
 'a/n': 'author note',
 'a/w': 'anyway',
 'a/s/l': 'age, sex, location',
 'adn': 'any day now',
 'afaic': "as far as i'm concerned",
 'afaik': 'as far as I know',
 'afk': 'away from keyboard',
 'aggro': 'aggresive',
 'aight': 'alright',
 'airhead': 'stupid',
 'aka': 'as known as',
 'alol': 'actually laughing out loud',
 'amigo': 'friend',
 'amz': 'amazing',
 'app': 'application',
 'armpit': 'undesirable',
 'asap': 'as soon as possible',
 'atm': 'at the moment',
 'atw': 'all the way',
 'b/c': 'because',
 'b-day': 'birthday',
 'b4': 'before',
 'b4n': 'bye for now',
 'bae': 'before anyone else',
 'bak': 'back at the keyboard',
 'bbl': 'bee back later',
 'bday': 'birthday',
 'becuz': 'because',
 'bent': 'angry',
 'bestie': 'best friend',
 'besty': 'best friend',
 'bf': 'boyfriend',
 'bff': 'best friends forever',
 'bffe': 'best friends forever',
 'bfn': 'bye for now',
 'bg': 'big grin',
 'bmfe': 'best mates forever',
 'bmfl': 'best mates life',
 'bozo': 'idiot',
 'brah': 'friend',
 'bravo': 'well done',
 'brb': 'be right back',
 'bro': 'brother',
 'bta': 'but then again',
 'btdt': 'been there, done that',
 'btr': 'better',
 'btw': 'by the way',
 'buddy': 'friend',
 "c'mon": 'came on',
 'cid crying in disgrace': '',
 'congrats congratulations': '',
 'copacetic excellent': '',
 'coz beacause': '',
 'cu': 'see you',
 'cuddy': 'friends',
 'cul': 'see you later',
 'cul8r': 'see you later',
 'cutie': 'cute',
 'cuz': 'because',
 'cya': 'bye',
 'cyo': 'see you online',
 'dbau': 'doing business as usual',
 'deets': 'details',
 'dmn': 'damn',
 'dobe': 'idiot',
 'dope': 'stupid',
 'dork': 'strange',
 'dunno': "don't know",
 'dwi': 'deal with it',
 'dyd': "don't you dare",
 'ermahgerd': 'oh my gosh',
 'eu': 'europe',
 'ez': 'easy',
 'f9': 'fine',
 'fav': 'favorite',
 'far-out': 'great',
 'fb': 'facebook',
 'flick': 'movie',
 'fml': 'fuck my life',
 'foxy': 'sexy',
 'friggin': 'freaking',
 'fttn': 'for the time being',
 'ftw': 'for the win',
 'fud': 'fear, uncertainty, and doubt',
 'fwiw': "for what it's worth",
 'fyi': 'for your information',
 'g': 'grin',
 'g2g': 'got to go',
 'ga': 'go ahead',
 'gal': 'get a life',
 'getcha': 'understand',
 'gf': 'girlfriend',
 'gfn': 'gone for now',
 'gg': 'good game',
 'gj': 'good job',
 'gky': 'go kill yourself',
 'gl': 'good luck',
 'glhf': 'good luck have fun',
 'gmab': 'give me a break',
 'gmbo': 'giggling my butt off',
 'gmta': 'great minds think alike',
 'goof': 'idiot',
 'goofy': 'idiot',
 'gr8': 'great',
 'gtg': 'got to go',
 'gud': 'good',
 'h8': 'hate',
 'hagn': 'have a good night',
 'hdop': 'help delete online predators',
 'hf': 'have fun',
 'hml': 'hate my life',
 'hoas': 'hold on a second',
 'hhis': 'hanging head in shame',
 'hmu': 'hit me up',
 'hru': 'how are you',
 'twt': 'hope this helps',
 'hw': 'homework',
 "i'ma": 'i am going to',
 'iac': 'in any case',
 'ic': 'I see',
 'icymi': 'in case you missed it',
 'idk': "I don't know",
 'iggy': 'ignore',
 'iht': 'i hate this',
 'ikr': 'i know, right?',
 'ilt': 'i like that',
 'ily': 'i love you',
 'ima': 'i am going to',
 'imao': 'in my arrogant opinion',
 'imnsho': 'in my not so humble opinion',
 'imo': 'in my opinion',
 'imy': 'i miss you',
 'iou': 'i owe you',
 'iow': 'in other words',
 'ipn': 'I’m posting naked',
 'irl': 'in real life',
 'j/k': 'just kidding',
 'jdi': 'just do it',
 'jk': 'just kidding',
 'jkn': 'joking',
 'jyeah': 'yeah',
 'kinda': 'kind of',
 'l8': 'late',
 'l8r': 'later',
 'lbh': "let's be honest",
 'ld': 'later, dude',
 'ldi': "let's do it",
 'ldr': 'long distance relationship',
 'lees': 'beautiful',
 'lfm': 'looking for more',
 'lil': 'little',
 'llta': 'lots and lots of thunderous applause',
 'lmao': 'laugh my ass off',
 'lmirl': "let's meet in real life",
 'lmk': 'let me know',
 'lol': 'laugh out loud',
 'lolz': 'laugh out loud',
 'lotta': 'lot of',
 'lsr': 'loser',
 'ltr': 'longterm relationship',
 'lua': 'love you always',
 'lub': 'love',
 'lubb': 'love',
 'lulab': 'love you like a brother',
 'lulas': 'love you like a sister',
 'lul': 'laugh',
 'luls': 'laugh',
 'lulz': 'laugh',
 'lumu': 'love you miss you',
 'luv': 'love',
 'lux': 'luxury',
 'lwm': 'laugh with me',
 'lwp': 'laugh with passion',
 'lvl': 'level',
 'm/f': 'male or female',
 'm2': 'me too',
 'm8': 'mate',
 'me2': 'me too',
 'milf': 'mother I would like to fuck',
 'mma': 'meet me at',
 'mmb': 'message me back',
 'mvp': 'most valueable player',
 'msg': 'message',
 'mtf': 'more to follow',
 'myob': 'mind your own business',
 'nah': 'no',
 'nc': 'no comment',
 'nk': 'not kidding',
 'ngl': 'not gonna lie',
 'nlt': 'no later than',
 'nm': 'not much',
 'no1': 'no one',
 'np': 'no problem',
 'nsfw': 'not safe for work',
 'nuh': 'no',
 'nvm': 'nevermind',
 'obo': 'or best offer',
 'oic': 'oh, i see',
 'oll': 'online love',
 'omg': 'oh my god',
 'omw': 'on my way',
 'osm': 'awesome',
 'otoh': 'on the other hand',
 'perv': 'pervert',
 'pervy': 'pervert',
 'phat': 'pretty hot and tempting',
 'pir': 'parent in room',
 'pls': 'please',
 'plz': 'please',
 'ppl': 'people',
 'pro': 'professional',
 'pwnd': 'owned',
 'qq': 'crying',
 'r': 'are',
 'rly': 'really',
 'rofl': 'roll on the floor laughing',
 'rolf': 'roll on the floor laughing',
 'rpg': 'role playing games',
 'ru': 'are you',
 's2u': 'shame to you',
 'scrub': 'loser',
 'sec': 'second',
 'shid': 'slaps head in disgust',
 'shoulda': 'should have',
 'sff': 'so funny',
 'smexy': 'smart and sexy',
 'smh': 'shaking my head',
 'somy': 'sick of me yet',
 'sot': 'short of time',
 'sry': 'sorry',
 'str8': 'straight',
 'sux': 'sucks',
 'swag': 'style',
 'taze': 'irritate',
 'tba': 'to be announced',
 'tbfu': 'too bad for you',
 'tbc': 'to be continued',
 'tbd': 'to be determined',
 'tbr': 'to be rude',
 'tc': 'take care',
 'thx': 'thanks',
 'thanx': 'thanks',
 'tfw': 'that feeling  when',
 'til': 'today i learned',
 'ttyl': 'talk to you later',
 'ty': 'thank you',
 'tyvm': 'thank you very much',
 'u': 'you',
 'uber': 'the best',
 'ugh': 'disgusted',
 'ur': 'you are',
 'uw': 'you are welcome',
 'vs': 'versus',
 'w2f': 'way too funny',
 'w8': 'wait',
 'wak': 'weird',
 'wanna': 'want to',
 'wb': 'welcome back',
 'whiz': 'talented',
 'whoa': 'surprise',
 'whoah': 'surprise',
 'wfm': 'works for me',
 'wibni': "wouldn't it be nice if",
 'wmd': 'weapon of mass destruction',
 'wot': 'what',
 'wtf': 'what the fuck',
 'wtg': 'way to go',
 'wtgp': 'want to go private',
 'wu': "what's up",
 'wuh': 'what?',
 'wuv': 'love',
 'ym': 'young man',
 'yawn': 'boring',
 'yum': 'good',
 'x': 'kiss',
 'xxx': 'kiss',
 'xdd': 'laughing', 
 'y': 'why',
 'yolo': 'you only live once',
 'yuge': 'huge',
 'yw': 'you are welcome',
 'ywa': 'you are welcome anyway',
 'zomg': 'oh my god!',
 'zzz': 'sleeping'
 }