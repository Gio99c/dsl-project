library(DiagrammeR)
library(DiagrammeRsvg)
library(magrittr)
library(rsvg)

text_mining <- grViz("
digraph neato {

graph [layout = neato]

node [shape = circle, fixedsize = true, width=0.7, peripheries=2, penwidth= 0.1, fontsize=10]
a [label = 'text']

node [shape = square, style=filled, fillcolor=grey, peripheries=1]
b [label = 'tf-df'] 
c [label = 'sentiment']
d [label = 'word\nembedding']

node [shape = circle, style = filled, fillcolor = white, peripheries=1]
e [label = 'neg']
f [label = 'neu']
g [label = 'pos']
h [label = 'compound']
i [label = 'polarity']
j [label = 'relevant\nwords\n...']
k [label = 'positivity\nlikelihood']
l [label = 'negativity\nlikelihood']

edge [color = grey]
a -> {b c d}
b -> {j}
c -> {e f g h i}
d -> {k l}
}")

text_mining %>%
  export_svg %>% charToRaw %>% rsvg_svg(file="./figures/text_schema.svg")
text_mining