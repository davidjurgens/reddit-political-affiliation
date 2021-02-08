DEM_PATTERN_SILVER = "((i am|i'm) a (democrat|liberal)|i vote[d]?( for| for a)? (democrat|hillary|biden|obama|blue))"

ANTI_REP_PATTERN_SILVER = "(i (hate|despise|loathe) (conservatives|republicans|trump|donald trump|mcconell|mitch mcconell)|" \
                          "(i am|i'm) a (former|ex) (conservative|republican)|(i am|i'm) an ex-(conservative|republican)|" \
                          "i (was|used to be|used to vote)( a| as a)? (conservative|republican)|" \
                          "fuck (conservatives|republicans|donald trump|trump|mcconell|mitch mcconell))"

REP_PATTERN_SILVER = "((i am|i'm) a (conservative|republican)|i vote[d]?( for| for a)? (" \
                     "republican|conservative|trump|romney|mcconell))"

ANTI_DEM_PATTERN_SILVER = "(i (hate|despise) (liberals|progressives|democrats|left-wing|biden|hillary|obama)|(i am|i'm) a (" \
                          "former|ex) (liberal|democrat|progressive)|(i am|i'm) an ex-(liberal|democrat|progressive)|i (" \
                          "was|used to be|used to vote)( a| as a)? (liberal|democrat|progressive)|fuck (" \
                          "liberals|progressives|democrats|biden|hillary|obama))"

# Gold standard only accepts direct declarations of affiliation

DEM_PATTERN_GOLD = "((i am|i'm) a (democrat|liberal)|i vote[d]?( for| for a)? (democrat|hillary|biden|obama|blue))"

ANTI_REP_PATTERN_GOLD = "(i am|i'm) a (former|ex) (conservative|republican)|(i am|i'm) an ex-(conservative|republican)|" \ 
                        "i (was|used to be|used to vote)( a| as a)? (conservative|republican)"

REP_PATTERN_GOLD = "((i am|i'm) a (conservative|republican)|i vote[d]?( for| for a)? (" \
                   "republican|conservative|trump|romney|mcconell))"

ANTI_DEM_PATTERN_GOLD = "(i am|i'm) a (" \
                        "former|ex) (liberal|democrat|progressive)|(i am|i'm) an ex-(liberal|democrat|progressive)|i (" \
                        "was|used to be|used to vote)( a| as a)? (liberal|democrat|progressive)"
