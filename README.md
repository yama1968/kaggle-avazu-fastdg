# kaggle-avazu-fastdg

This is a repository for the Good Timers Python implementation of Google FTRL Proximal and Microsoft AdPredictor.

For more information, refer to http://bit.ly/CTRK1 and http://bit.ly/CBLOGBOOK1.

# Instructions

You need to get the data from the Avazu CTR contest on Kaggle, or to use the small sample provided, small3.train.csv.

Please refer to the Makefile, and to the help from:

    python fastdg_v4_5.py -h

You can also test AdPredictor with:

    make small3.q02db.fg4.model.gz small3.q02db.test

And FTRL Proximal with:

    make small3.g02a.fg4.model.gz small3.g02a.test

You must have Python 2.7 installed with pypy.

# Disclaimer

You use this program from your own choice. You can refer to tinrtgu and to me if you want to be nice. Sorry, the code and options are not really tidied up yet :-).

# Credits

Thanks to Christophe Bourguignat, my dear partner for this challenge.

Thanks to tinrtgu for his http://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory, and for providing a Python implementation of FTRL Proximal, on which I have based the script presented here.

Thanks to the team at Microsoft Research for AdPredictor, and as well to the Google FTRL team!
