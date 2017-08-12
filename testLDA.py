from Helper import Helper
from LDAModel import LDAModel
import pandas as pd

if __name__ == '__main__':
    helper = Helper()
    my_papers = \
        "Cascading failure tolerance of modular small-world networks" + \
        "Region-Based analysis of hybrid petri nets with a single general one-shot transition" + \
        "Survivability Evaluation of Gas, Water and Electricity Infrastructures" + \
        "Survivability evaluation of fluid critical infrastructures using hybrid Petri nets" + \
        "Energy resilience modelling for smart houses" + \
        "Survivability analysis of a sewage treatment facility using hybrid Petri nets" + \
        "Fluid Survival Tool: A Model Checker for Hybrid Petri Nets" + \
        "Hybrid Petri nets with multiple stochastic transition firings" + \
        "Approximate Analysis of Hybrid Petri Nets with Probabilistic Timed Transitions" + \
        "Pricing in population games with semi-rational agents" + \
        "Analysis of hybrid Petri nets with random discrete events" + \
        "Influence of short cycles on the PageRank distribution in scale-free random graphs"

    my_word_list = helper.preprocess_text(my_papers).split()

    lda_model = LDAModel()
    my_topics = lda_model.extract_topics(my_word_list)
    my_topics_df = pd.DataFrame(my_topics, columns=['id', 'prob']).sort_values('prob', ascending=False)
    my_topics_df['topic'] = my_topics_df['id'].apply(lda_model.ldamodel.print_topic)
    print (my_topics_df)
