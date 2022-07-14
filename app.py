
# # # install dependecies
# pip install PyPDF2
# pip install SentencePiece
# pip install transformers



import streamlit as st
import os
import PyPDF2
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from io import StringIO 


url_pegasus = 'https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html'
st.markdown("# PDF summarization using Pegasus")
st.markdown(
  '''PhD scholars are often tasked with reading a Scientifc paper and producing a short summary of it to demonstrate both reading 
  comprehension and writing ability. This is called abstractive summarization where summaries are based on understanding of the whole 
  passage and ability to rephrase it and this is one of the most difficult task in NLP. It was not until the development of techniques 
  like seq2seq learning and unsupervised language models (e.g., ELMo and BERT) that abstractive summarization becomes more feasible.
Building upon earlier breakthroughs in natural language processing (NLP) field, Google’s [PEGASUS](%s) further improved the state-of-the-art (SOTA) 
results for abstractive summarization, in particular with low resources. To be more specific, unlike previous models, PEGASUS enables us to achieve
 close to SOTA results with 1,000 examples, rather than tens of thousands of training data. The model uses Transformers Encoder-Decoder architecture. 
 The encoder outputs masked tokens while the decoder 
  generates Gap sentences. ''' %url_pegasus)



st.header(''' In this PDF summarizer you can upload any PDF of size less than 200MB using the Browse file button and can get its abstractoive summary within minutes thoug the speed will depend on the number of text in the PDF. Note: Upload only one PDF at a time.''')

uploaded_file = st.file_uploader("Upload a file")

if uploaded_file:

  with open(os.path.join(uploaded_file.name),"wb") as f: 
        f.write(uploaded_file.getbuffer())         
  st.success("Saved File")


st.markdown("# Summary")


# # Load tokenizer 
# length_summary = st.slider(
#   'Choose the length of summary:',
#   min_value= 1,max_value=1000)
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

# Load model 
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")



# st.spinner('Wait for it...')
#   # st.markdown("Loading. Please wait")
# st.success('Done!')





if uploaded_file:
    fhandle = open(uploaded_file.name, 'rb')
    pdfReader = PyPDF2.PdfFileReader(fhandle)

    values = st.slider(
     'Summarize page number between:',
     min_value= 1, max_value= pdfReader.numPages,
     value= (1,pdfReader.numPages-1 ),
     step = 1)


    with st.spinner('Preparing summary. Please wait...'):
      for x in range(values[0]-1,values[1]-1):
        
        page_wise_text = []
        # st.write("get page x: ", x)
        pagehandle = pdfReader.getPage(x)
        # page_wise_text.append('Page number: {}'.format(x+1))
        text = pagehandle.extractText()

        # Create tokens - number representation of our text
        tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")

        # Summarize 
        
        summary = model.generate(**tokens)

        # Decode summary
        decoded_output = tokenizer.decode(summary[0])

        # update summary on the list
        # page_wise_text.append("Page number: {}".format(x+1))
        page_wise_text.append(decoded_output)
        
        # page_wise_text.append("Page Break")

        # print summary

        st.markdown("Page number: {}".format(x+1))
        st.text_area(label ="",value=page_wise_text, placeholder="Please upload a PDF to get it's summary", height = 100, key=x+1)
        st.markdown("")

      # st.write(page_wise_text)
      #st.text_area(label ="",value=page_wise_text, placeholder="Please upload a PDF to get it's summary", height = 500)

