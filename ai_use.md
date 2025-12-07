# Process of AI Usage in This Project:
This was my first time coding in concert with an AI tool. I've typically resisted for fear of over reliance, but in this project I wanted to expirement with applying AI to explore the field of Data Science in Python. After all, while I've used R Studio in AQM1000, 2000 and QTM 3635 which has given me a basic understanding of data science and the various processes I might need to do. I used AI to help bridge the language gap to convert some of my data premonitions to Python code and helpin me understand how Python accomplishes similar code to what I've seen in R. 

I used Claude Code as my copilot of choice. I kept Claude in the "Ask Before Edits" so I could closely monitor any code generation. This helped me learn and make stylistic adjustments as I progressed through the project. Due to the odd structure of my data and some of the contextual awareness that Data Science requires, this was a worth while decision. 

I will say as I progressed through the project my AI Use increased alongside my comfort with interpretting/using Claude. That said, in my README I've added a comprehensive "Code Explantaion" section to demonstrate my understanding of the various files. 

## Heaviest Usage:

### OpenAI Request Parallelization
Due to the time constraints of this project I knew I would need to convert my synchronous OpenAI API calls to Async functions and parallellize my requests. As such I first built a POC of my GPT enrichment script (almost all by hand) using the sync functions. This helped me understand the basic workflow that would need to occur at each row. I made some initial requests to verify this process.  

Converting to the parallel process was heavily Claude augmented. That said, I wanted to be cognizant of my token usage, potential rate limiting and ensure any programs Claude wrote were explicable + able to be tested partially before letting lose across all 1000 rows. 

To prevent and manage possible complications I did two things:
1. Spec driven development for the conversion. I was very explicit in my specs for the new parallelized function. This included specifiying the number of parallel instances I'd like to be able to spin up, the abillity to test partially, 'save' progress every 50 lines and to have the function check for whether the record had already been enriched. 
2. Once this was built I reviewed the code (even phoning a friend who is a SWE to help understand the nuances of something like a semaphore). Then I ran the script on a limited portion of the CSV to verify the code worked like I expected without wastefully burning through tokens. 


### EDA
The EDA notebook was where I used the most of Claude's functionality. I used Claude to help me learn the breadth of Pandas' functions. This helped me convert aspects of data manipulation that I intrinsically understood into Python code. These prompts tended to be fairly explicit. An example is shown below.

I will say that this accelerated my process of understanding some Pandas functions as when I coded the data manipulations in my modelling script I was able to recall what Claude had demonstrated in the EDA. 

#### Sample Prompt:
>I want to create another cell under cell 31 redoing the same brand distribution analysis but after combining Jordan Brand with Jordan and Air Jordan. And then creating an Other category for all the brands with less than 5 records


**Note: all markdown cells in the EDA notebook were handwritten and they are intended to show some of my thoughts as I progressed through the EDA**

### Modelling:
Similar to the EDA step I used Claude to help me build the code for the neural network with what I know about model architecture from Davit Katchatryan's ML course. That said I am proud to say (and somewhat dissapointed in Claude) I noticed a few structural errors in Claude's code. 

For example instead of properly reccomending a K-fold cross validation process to optimize across hyperparamters in the network architecture and dropout Claude CVed across training/test splits. Which while possible isn't as streamlined or optimal as optimizing hyperparamters like the number of hidden layers and dropout. I noticed this as it was antagonistic to what I've done in R with H2O. 

An excerpt from Claude when I caught this error:
> Great catch! You're absolutely right to be confused - I made an error in my explanation. What we built earlier: The train_model_kfold function was doing K-fold CV with fixed hyperparameters (hidden_size=128, dropout=0.3, lr=0.001). It was just training the same model 5 times on different data splits to get a more robust estimate of performance.

Of all the code in this repo the modelling is the one part where the code began to feel (for lack of a better word) convoluted. Thus, I want to revist the PyTorch and Scikit documentation and try rebuilding another NN across more records. I also have an RTX-2070 which I would like to expirement with using for training, at some point. Alas, this will have to be dogeared for another day.

### Importing the models into Streamlit
I used AI to help outline the skeleton of the streamlit app and to assist with the code required to carry over the Neural Network from model1.py. That said, I implemented most other functions myself and did the plumbing to re-call functions used in other Python modules. 

## Meta Prompting
Since I also made OpenAI API calls within my processing scripts I metaprompted to create a very clear prompt that I could pass into the OpenAI API for the best results. 


### Meta Prompt to Create [Prompt1](src/models/prompts/prompt1.py)
> I will be passing an article about sneakers into an OpenAI api call as a string. These articles are typically around ~300 words but can range up to 1,500 on rare occasion. I’m planning on using GPT to add features to a training set for an ML model I’m making downstream  From this string I want to pull: 
>
> Sneaker_price: should only be a single value $ amount of the primary sneaker being mentioned. If there are multiple prices mentioned, deem to pick the highest one
>
> Classify the article type into one of the three following buckets:
> - Sneaker releases: Details information about an upcoming sneaker release
> - News: Sneaker related news, such as a news article on Kanye West's lawsuit with adidas 
> - Features: typically “listicles” like top 10 most
>
> Sneaker_brand: The company of which the primary sneaker discussed in this article is made by. Should be only one brand returned and if no brand is mentioned an empty string.  brand returned and if no brand is mentioned an empty string. 

