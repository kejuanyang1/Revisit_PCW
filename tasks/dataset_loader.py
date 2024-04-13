import logging
from abc import ABC
from typing import Dict, Optional

import os
import pandas as pd
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

DATA_DIR = '<your data path>'
EXAMPLE_PROMPT_PATH = 'tasks/prompts'

SPLIT_TOKEN = "\n"
TEXT_BETWEEN_SHOTS = f"\n{SPLIT_TOKEN}\n"
TEXT_BETWEEN_SHOTS_CLASS = f"\n==\n"
TEXT_BETWEEN_SHOTS_CLASS_GLM = f"\n\n\n"

# df columns
N_TOKENS = 'n_tokens'
PROMPTS = 'prompts'
LABEL_TOKENS = 'label_tokens'

# input prompt
UTTERANCE_PREFIX = 'utterance: '
INTENT_PREFIX = 'intent: '


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


class ClassDS(Dataset):
    def __init__(self, df):
        self.text = df[PROMPTS].tolist()
        self.label = df[LABEL_TOKENS].tolist()

    def __getitem__(self, index):
        text = TEXT_BETWEEN_SHOTS_CLASS + self.text[index]
        assert text == text.rstrip(), "prompt ends with a space!"
        label = self.label[index]
        return text, label

    def __len__(self):
        return len(self.text)


class GenerationDS(Dataset):
    def __init__(self, df, instruction):
        self.text = df[PROMPTS].tolist()
        self.label = df[LABEL_TOKENS].tolist()
        self.instruction = instruction

    def __getitem__(self, index):
        text = self.instruction + self.text[index]
        assert text == text.rstrip(), "prompt ends with a space!"
        label = self.label[index]
        return text, label

    def __len__(self):
        return len(self.text)


class GenerationDatasetAccess(ABC):
    name: str
    dataset: Optional[str] = None
    x_column: str = 'question'
    y_label: str = 'answer'
    x_prefix: str = "Question: "
    y_prefix: str = "Answer: "
    example_prompt: str = ""
    instruction: str = ""

    def __init__(self):
        super().__init__()
        if self.dataset is None:
            self.dataset = self.name
        self.data_dir = DATA_DIR
        if self.example_prompt != "":
            test_dataset = self._load_dataset()
            example_df = pd.read_csv(f'{EXAMPLE_PROMPT_PATH}/{self.example_prompt}.csv')
        else:
            train_dataset, test_dataset = self._load_dataset()
            example_df = train_dataset.to_pandas()
            if len(example_df) > 15000:
                example_df = example_df.iloc[:15000]
        test_df = test_dataset.to_pandas()
        _logger.info(f"loaded {len(test_df)} test samples")
        self.test_df = self.apply_format(test_df, test=True)
        self.example_df = self.apply_format(example_df)

    def _load_dataset(self):
        if '/' in self.dataset:
            self.dataset = self.dataset.split('/')[-1]
        hf_filepath = os.path.join(self.data_dir, self.dataset)
        dataset = load_from_disk(hf_filepath)
        if self.example_prompt != "":
            return dataset
        if 'validation' in dataset:
            return dataset['train'], dataset['validation']
        if 'test' not in dataset:
            _logger.error("no test split!")
        return dataset['train'], dataset['test']

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_y_answer(self, df):
        df[LABEL_TOKENS] = df[self.y_label]
        return df

    def apply_format(self, df, test=False):
        df = self.generate_x_text(df)
        df = self.generate_y_answer(df)
        if test:
            df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.y_prefix}".rstrip(), axis=1)
        else:
            df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.y_prefix}{x[LABEL_TOKENS]}",
                                   axis=1)
        return df
    

class COTGenerationDatasetAccess(GenerationDatasetAccess):
    t_column: str = "thought"
    t_prefix: str = "Let's think step by step. "

    def apply_format(self, df, test=False):
        df = self.generate_x_text(df)
        df = self.generate_y_answer(df)
        if test:
            df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.t_prefix}".rstrip(), axis=1)
        else:
            df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.t_prefix}{x[self.t_column]}\n{self.y_prefix}{x[LABEL_TOKENS]}",
                                   axis=1)
        return df


class ClassificationDatasetAccess(ABC):
    name: str
    dataset: Optional[str] = None
    subset: Optional[str] = None
    x_column: str = 'text'
    y_label: str = 'label'
    x_prefix: str = "Review: "
    y_prefix: str = "Sentiment: "
    label_mapping: Optional[Dict] = None
    map_labels: bool = True

    def __init__(self):
        super().__init__()
        if self.dataset is None:
            self.dataset = self.name
        self.data_dir = DATA_DIR
        train_dataset, test_dataset = self._load_dataset()
        train_df = train_dataset.to_pandas()
        test_df = test_dataset.to_pandas()
        _logger.info(f"loaded {len(train_df)} training samples & {len(test_df)} test samples")

        if self.map_labels:
            hf_default_labels = train_dataset.features[self.y_label]
            default_label_mapping = dict(enumerate(hf_default_labels.names)) if hasattr(
                train_dataset.features[self.y_label], 'names') else None
            self._initialize_label_mapping(default_label_mapping)

        self.train_df = self.apply_format(train_df)
        self.test_df = self.apply_format(test_df, test=True)

    def _initialize_label_mapping(self, default_label_mapping):
        if self.label_mapping:
            _logger.info("overriding default label mapping")
            if default_label_mapping:
                _logger.info([f"{default_label_mapping[k]} -> "
                              f"{self.label_mapping[k]}" for k in self.label_mapping.keys()])
        else:
            _logger.info(f"using default label mapping: {default_label_mapping}")
            self.label_mapping = default_label_mapping

    def _load_dataset(self):
        if self.subset is not None:
            self.dataset = self.subset
        if '/' in self.dataset:
            self.dataset = self.dataset.split('/')[-1]

        hf_filepath = os.path.join(self.data_dir, self.dataset)
        dataset = load_from_disk(hf_filepath)
        if 'validation' in dataset:
            return dataset['train'], dataset['validation']
        if 'test' not in dataset:
            _logger.info("no test or validation found, splitting train set instead")
            dataset = dataset['train'].train_test_split(seed=42)

        return dataset['train'], dataset['test']

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_y_token_labels(self, df, test):
        if self.map_labels:
            df[LABEL_TOKENS] = df[self.y_label].map(self.label_mapping)
        else:
            df[LABEL_TOKENS] = df[self.y_label]
        return df

    @property
    def labels(self):
        if self.map_labels:
            return self.label_mapping.values()
        else:
            return self.test_df[LABEL_TOKENS].unique()

    def apply_format(self, df, test=False):
        df = self.generate_x_text(df)
        df = self.generate_y_token_labels(df, test)
        if test:
            df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.y_prefix}".rstrip(), axis=1)
        else:
            df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.y_prefix}{x[LABEL_TOKENS]}",
                                   axis=1)
        return df


class SST5(ClassificationDatasetAccess):
    name = 'sst5'
    dataset = 'SetFit/sst5'
    label_mapping = {0: 'terrible', 1: 'bad', 2: 'okay', 3: 'good', 4: 'great'}


class RTE(ClassificationDatasetAccess):
    name = 'rte'
    dataset = 'super_glue'
    subset = 'rte'
    x_prefix = ''
    y_prefix = 'prediction: '
    label_mapping = {0: 'True', 1: 'False'}

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df.apply(lambda x: f"premise: {x['premise']}\nhypothesis: {x['hypothesis']}", axis=1)
        return df


class CB(RTE):
    name = 'cb'
    subset = 'cb'
    label_mapping = {0: 'true', 1: 'false', 2: 'neither'}


class SUBJ(ClassificationDatasetAccess):
    name = 'subj'
    dataset = 'SetFit/subj'
    label_mapping = {0: 'objective', 1: 'subjective'}
    x_prefix = 'Input: '
    y_prefix = 'Type: '


class CR(ClassificationDatasetAccess):
    name = 'cr'
    dataset = 'SetFit/CR'
    label_mapping = {0: 'negative', 1: 'positive'}


class AGNEWS(ClassificationDatasetAccess):
    name = 'agnews'
    dataset = 'ag_news'
    label_mapping = {0: 'world', 1: 'sports', 2: 'business', 3: 'technology'}
    x_prefix = 'input: '
    y_prefix = 'type: '


class DBPEDIA(ClassificationDatasetAccess):
    name = 'dbpedia'
    # dataset = 'dbpedia_14'
    label_mapping = {0: 'company',
                     1: 'school',
                     2: 'artist',
                     3: 'athlete',
                     4: 'politics',
                     5: 'transportation',
                     6: 'building',
                     7: 'nature',
                     8: 'village',
                     9: 'animal',
                     10: 'plant',
                     11: 'album',
                     12: 'film',
                     13: 'book'}
    x_prefix = 'input: '
    y_prefix = 'type: '

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['content']
        return df


class SST2(ClassificationDatasetAccess):
    name = 'sst2'

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['sentence']
        return df


class TREC(ClassificationDatasetAccess):
    name = 'trec'
    y_label = 'coarse_label'
    x_prefix = "Question: "
    y_prefix = "Type: "
    label_mapping = {0: "abbreviation", 1: "entity", 2: "description", 3: "human", 4: "location", 5: 'numeric'}


class TRECFINE(ClassificationDatasetAccess):
    name = 'trecfine'
    dataset = 'trec_fine'
    y_label = 'fine_label'
    x_prefix = "Question: "
    y_prefix = "Type: "
    # labels mapping based on: https://aclanthology.org/C16-1116.pdf, https://aclanthology.org/C02-1150.pdf
    label_mapping = {0: 'abbreviation abbreviation',
                     1: 'abbreviation expansion',
                     2: 'entity animal',
                     3: 'entity body',
                     4: 'entity color',
                     5: 'entity creation',
                     6: 'entity currency',
                     7: 'entity disease',
                     8: 'entity event',
                     9: 'entity food',
                     10: 'entity instrument',
                     11: 'entity language',
                     12: 'entity letter',
                     13: 'entity other',
                     14: 'entity plant',
                     15: 'entity product',
                     16: 'entity religion',
                     17: 'entity sport',
                     18: 'entity substance',
                     19: 'entity symbol',
                     20: 'entity technique',
                     21: 'entity term',
                     22: 'entity vehicle',
                     23: 'entity word',
                     24: 'description definition',
                     25: 'description description',
                     26: 'description manner',
                     27: 'description reason',
                     28: 'human group',
                     29: 'human individual',
                     30: 'human title',
                     31: 'human description',
                     32: 'location city',
                     33: 'location country',
                     34: 'location mountain',
                     35: 'location other',
                     36: 'location state',
                     37: 'numeric code',
                     38: 'numeric count',
                     39: 'numeric date',
                     40: 'numeric distance',
                     41: 'numeric money',
                     42: 'numeric order',
                     43: 'numeric other',
                     44: 'numeric period',
                     45: 'numeric percent',
                     46: 'numeric speed',
                     47: 'numeric temperature',
                     48: 'numeric size',
                     49: 'numeric weight'}


class YELP(ClassificationDatasetAccess):
    name = 'yelp'
    dataset = 'yelp_review_full'
    x_prefix = 'review: '
    y_prefix = 'stars: '
    label_mapping = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}


class BANKING77(ClassificationDatasetAccess):
    name = 'banking77'
    x_prefix = 'query: '
    y_prefix = INTENT_PREFIX

    def _initialize_label_mapping(self, default_label_mapping):
        default_label_mapping = {k: v.replace('_', ' ') for k, v in default_label_mapping.items()}
        super()._initialize_label_mapping(default_label_mapping)


class NLU(ClassificationDatasetAccess):
    name = 'nlu_intent'
    # dataset = 'nlu_evaluation_data'
    x_prefix = UTTERANCE_PREFIX
    y_prefix = INTENT_PREFIX
    label_mapping = {0: 'alarm query', 1: 'alarm remove', 2: 'alarm set', 3: 'audio volume down',
                     4: 'audio volume mute', 5: 'audio volume other', 6: 'audio volume up', 7: 'calendar query',
                     8: 'calendar remove', 9: 'calendar set', 10: 'cooking query', 11: 'cooking recipe',
                     12: 'datetime convert', 13: 'datetime query', 14: 'email add contact', 15: 'email query',
                     16: 'email query contact', 17: 'email sendemail', 18: 'general affirm', 19: 'general command stop',
                     20: 'general confirm', 21: 'general dont care', 22: 'general explain', 23: 'general greet',
                     24: 'general joke', 25: 'general negate', 26: 'general praise', 27: 'general quirky',
                     28: 'general repeat', 29: 'iot cleaning', 30: 'iot coffee', 31: 'iot hue light change',
                     32: 'iot hue light dim', 33: 'iot hue light off', 34: 'iot hue lighton', 35: 'iot hue light up',
                     36: 'iot wemo off', 37: 'iot wemo on', 38: 'lists create or add', 39: 'lists query',
                     40: 'lists remove', 41: 'music dislikeness', 42: 'music likeness', 43: 'music query',
                     44: 'music settings', 45: 'news query', 46: 'play audiobook', 47: 'play game', 48: 'play music',
                     49: 'play podcasts', 50: 'play radio', 51: 'qa currency', 52: 'qa definition', 53: 'qa factoid',
                     54: 'qa maths', 55: 'qa stock', 56: 'recommendation events', 57: 'recommendation locations',
                     58: 'recommendation movies', 59: 'social post', 60: 'social query', 61: 'takeaway order',
                     62: 'takeaway query', 63: 'transport query', 64: 'transport taxi', 65: 'transport ticket',
                     66: 'transport traffic', 67: 'weather query'}


class NLUSCENARIO(ClassificationDatasetAccess):
    name = 'nlu_scenario'
    # dataset = 'nlu_evaluation_data'
    x_prefix = UTTERANCE_PREFIX
    y_prefix = 'scenario: '
    y_label = 'scenario'
    map_labels = False


class CLINIC150(ClassificationDatasetAccess):
    name = "clinic"
    # dataset = 'clinc_oos'
    # subset = 'plus'
    # y_label = "label_text"
    x_prefix = UTTERANCE_PREFIX
    y_prefix = INTENT_PREFIX
    label_mapping = {0: 'direct deposit', 1: 'carry on', 2: 'whisper mode', 3: 'text', 4: 'recipe', 5: 'smart home', 6: 'who do you work for', 7: 'rewards balance', 8: 'restaurant reservation', 9: 'travel notification', 10: 'update playlist', 11: 'change volume', 12: 'routing', 13: 'mpg', 14: 'bill balance', 15: 'do you have pets', 16: 'cook time', 17: 'what song', 18: 'new card', 19: 'todo list update', 20: 'traffic', 21: 'next song', 22: 'where are you from', 23: 'tire change', 24: 'bill due', 25: 'greeting', 26: 'taxes', 27: 'lost luggage', 28: 'change accent', 29: 'todo list', 30: 'last maintenance', 31: 'make call', 32: 'gas type', 33: 'cancel reservation', 34: 'schedule meeting', 35: 'find phone', 36: 'insurance change', 37: 'improve credit score', 38: 'travel suggestion', 39: 'roll dice', 40: 'repeat', 41: 'play music', 42: 'are you a bot', 43: 'sync device', 44: 'calendar', 45: 'insurance', 46: 'international visa', 47: 'freeze account', 48: 'shopping list', 49: 'oil change when', 50: 'share location', 51: 'what can i ask you', 52: 'plug type', 53: 'vaccines', 54: 'payday', 55: 'application status', 56: 'next holiday', 57: 'tell joke', 58: 'ingredient substitution', 59: 'calendar update', 60: 'how old are you', 61: 'directions', 62: 'definition', 63: 'rollover 401k', 64: 'pto request status', 65: 'confirm reservation', 66: 'expiration date', 67: 'calories', 68: 'timer', 69: 'transfer', 70: 'book flight', 71: 'change ai name', 72: 'apr', 73: 'accept reservations', 74: 'exchange rate', 75: 'pay bill', 76: 'weather', 77: 'current location', 78: 'cancel', 79: 'restaurant reviews', 80: 'pin change', 81: 'account blocked', 82: 'what are your hobbies', 83: 'oil change how', 84: 'reminder update', 85: 'car rental', 86: 'pto balance', 87: 'translate', 88: 'user name', 89: 'how busy', 90: 'yes', 91: 'replacement card duration', 92: 'what is your name', 93: 'gas', 94: 'tire pressure', 95: 'thank you', 96: 'pto request', 97: 'meal suggestion', 98: 'fun fact', 99: 'nutrition info', 100: 'card declined', 101: 'ingredients list', 102: 'distance', 103: 'book hotel', 104: 'travel alert', 105: 'damaged card', 106: 'flip coin', 107: 'restaurant suggestion', 108: 'min payment', 109: 'balance', 110: 'measurement conversion', 111: 'w2', 112: 'uber', 113: 'shopping list update', 114: 'change user name', 115: 'calculator', 116: 'order status', 117: 'food last', 118: 'reset settings', 119: 'meeting schedule', 120: 'timezone', 121: 'order checks', 122: 'spending history', 123: 'report fraud', 124: 'jump start', 125: 'no', 126: 'who made you', 127: 'interest rate', 128: 'report lost card', 129: 'international fees', 130: 'income', 131: 'reminder', 132: 'change speed', 133: 'redeem rewards', 134: 'change language', 135: 'transactions', 136: 'schedule maintenance', 137: 'date', 138: 'pto used', 139: 'spelling', 140: 'meaning of life', 141: 'flight status', 142: 'credit limit', 143: 'maybe', 144: 'alarm', 145: 'credit score', 146: 'goodbye', 147: 'time', 148: 'order', 149: 'credit limit change'}

class HotpotCOT(COTGenerationDatasetAccess):
    name = 'hotpotqa'
    example_prompt = "filter_cot18"
    instruction = "Solve a question answering task using your knowledge by reasoning continuously.\n" #Above are some examples.



DATASET_NAMES2LOADERS = {'sst5': SST5, 'sst2': SST2, 'agnews': AGNEWS, 'dbpedia': DBPEDIA, 'trec': TREC, 'cr': CR,
                         'cb': CB, 'rte': RTE, 'subj': SUBJ, 'yelp': YELP, 'banking77': BANKING77,
                         'nlu': NLU, 'nluscenario': NLUSCENARIO, 'trecfine': TRECFINE, 'clinic150': CLINIC150, 
                         'hotpotcot': HotpotCOT, }                  

if __name__ == '__main__':
    for ds_name, da in DATASET_NAMES2LOADERS.items():
        _logger.info(ds_name)
        _logger.info(da().example_df[PROMPTS].iloc[0])
        # _logger.info(da().test_df[PROMPTS].iloc[0])
