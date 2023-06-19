# Simplified version of NewBing 

This program is a simplified version of NewBing, feeding a question and relative news to OpenAI model as prompt to get the answer.

## Prerequisites package:
- openai
- newsdataapi
- spacy
- sentence_transformers
- sklearn
- retrying
- argparse
- json
- langdetect
- googletrans




Make sure you have those packages ready before running the script. You can install them by this terminal command:


···

pip install openai json spacy sentence_transformers sklearn retrying argparse langdetect googletrans==4.0.0-rc1

···



Please also specify your OpenAI API key and the question you want to ask in the run.sh. The default question is "Who is the US president?"

## Run Script
To run the script, use the following commend:


···

./run.sh

···


If permission got denied, type the following command and try running it again:



···

chmod +x run.sh

···


## Output
The output will be the question and the answer. It will present the source of the news if there is relative news found.


Feel free to contact jundong0526@gmail.com if you have any issue.