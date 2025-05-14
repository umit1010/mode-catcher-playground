#### NLP FUNCTIONS SUB-MODULE
####  Umit created this overly simple module file on 04/21/2025
####  to move NLP logic to a separate file so that when we make changes
####  only this file would need to be pushed to git & we would avoid conflicts
####  which inevitably happen quite often with our current workflow
####  of just having a single app.py file :)

#### Note on 04/21: I could not move most of the functions here yet because
#### we use globals a lot in our original code. I will do so when
#### I move those globals to dcc.Store objects.


def has_excluded_nlp_tag(token):

    # Parts of speech tags that should be automatically excluded
    # UH (3252815442139690129) == Interjection
    # IN (1292078113972184607) == Preposition

    # Dependency tags that should be automatically excluded
    # intj (421) == interjection
    # prep (443) == preposition
    # mark (423) == marker
    # acomp (398) = "adjectival complement"
    # parataxis (436)

    return (token.tag == 3252815442139690129 or token.tag == 1292078113972184607 or token.dep == 421 or token.dep == 423
        or token.dep == 398 or token.dep == 436)
