from transformers import BertModel, BertTokenizer
import pandas as pd


def make_dataset(df,padding_size=28):
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	dataset_df=pd.DataFrame()
	dataset_df[0]=df[0].apply(lambda x: tokenizer(x, padding='max_length',max_length=padding_size)["input_ids"])
	dataset_df[1]=df[1].apply(lambda x: np.array(float(x)))
	return(dataset_df)

