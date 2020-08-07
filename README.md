## Understanding sentiment of various vehicle makes using sentiment analysis

Edmunds.com is an online resource for automotive information. The website includes data such as prices for new and used vehicles, dealer listings, a directory of incentives and rebates. Edmunds.com also allows individuals to review vehicles on the website. The OpinRank Review Dataset has compiled over 42,000 such reviews of cars and trucks of various makes and models from 2007 to 2009. 

The dataset was retrieved from the OpinRank Review Dataset, which totaled more than 42,000 car reviews. Thirty makes, or manufacturers, were represented, including Toyota, Ford, Chevrolet and Honda. Smaller brands, such as Smart and Suzuki, were also included. Acura, Lexus, Mercedes-Benz, and BMW were some of the luxury brands represented. 

Each review was placed into a text file corresponding to the year, make and model of the vehicle to which it related. Each make contained a varying number of reviews. Those values are listed in figure 1.

The reviews were imported into a dataframe with the features “year”, “make”, “model” and “review” using regular expressions to filter out the various delimiters within the text files. Stop words were then filtered out and the reviews were tokenized for each make and model for future analysis.

Various plots, such as sentiment vs. market share plots, word clouds, and plots illustrating average sentiment by make are included.

Cars.zip holds the raw review data, and Hogue_Final_Project.py is the primary script while share_scrape.py creates the market share data.
