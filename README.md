# Literature Review Assistant
**Goal:** this notebook provides an LLM "research assistant" that can help with exploring the vaguest of questions by providing the relevant papers on the matter. It does not read the papers for you, but it provides a short summary that can be more helpful than the standard abstract when checking out several papers in one sitting. 

**Secondary Goal:** this "assistant" helps the exploration of sub-topics and related keywords; this is helpful when the question is vague or lacks crucial keywords, when the topic is vast, or when you want to look at things for yourself in other resources. 

The data source used for papers here is arXiv (this project is not affiliated with arXiv, or Google, in any way) - thank you to arXiv for use of its open access interoperability! 


## Some Q&A

### Who is this tool for? 
Anyone who wants to deepen their understanding of any topic that is published on arXiv by reading some papers! Hopefully, it can be useful to students and researchers, as well as those who are willing to tackle academic writing without going to school. 

### Why doesn't it read for me by default? 
LLMs can create a summary of a text; that's true. Reading scientific papers can sometimes be about reaching a concise review of methods and results, or figuring out an overview for a concensus. For this kind of research, using a tool along the lines of [elicit](https://elicit.com) can be helpful. 

Oftentimes, the reading process is about having a "conversation" of sorts with the author: do you agree with these methods, theories, interpretations? is there anything in this work that conflicts with or sheds a new light on what you already know? how does it apply to your own work? a summary won't answer these questions as well as reading the paper would. 


### How does this work?
The way this notebook works is best described in [a diagram](https://www.figma.com/board/pYy7LkzgMSOf20A6KX1ql8/AI-Literary-Review-Assistant?node-id=0-1&t=7ZskbjkvqklOHlSK-1). 
Baically, the LLM generates several arXiv API calls to answer the user request; the results from those results are stored in a vector database to perform RAG. These results are then summarized by the LLM, providing a concise list of reading material for the user. 

### Is the RAG really necessary here? 
Well, technically, no - the amount of data is very small (50 paper summaries), so using any embedding system (say, Doc2Vec) with a Pandas/Polars DataFrame would be enough; I still used ChromeDB as it allows for a seperate embedding of the user request as an intent that fits into the vector space created by the document embeddings. That is something that primitive solutions cannot provide, and it valuable to this kind of task (although I did not test the alternate solutions). 

### What's the point in RAG retreival if arXiv already does the ranking? 
The goal is to improve upon the arXiv results using the benefits of advanced embedding techniques, allowing for research queries that would rarely produce something helpful in classic search engines. The #1 arXiv result to the best keywords the LLM could provide is shown in the results, but so are some other articles that are a great match for the specific topic at hand, which may appear in later pages or for realted keywrods. It could be the case that more accurate usage of keywrods on arXiv would provide the exact same results as this tool; but the aim here is to save the time of figuring these nuances out. 
