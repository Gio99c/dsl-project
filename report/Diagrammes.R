library(DiagrammeR)
library(DiagrammeRsvg)
library(magrittr)
library(rsvg)

text_schema <- grViz("
digraph neato {

graph [layout = neato]

node [shape = circle, fixedsize = true, width=0.7, peripheries=2, penwidth= 0.5, fontsize=10]
a [label = 'Text']

node [shape = square, style=filled, fillcolor='#80b1d3', peripheries=1]
b [label = 'TF-DF'] 
c [label = 'Sentiment']
d [label = 'Word\nEmbedding']

node [shape = circle, style = filled, fillcolor = white, peripheries=1]
e [label = 'neg']
f [label = 'neu']
g [label = 'pos']
h [label = 'compound']
i [label = 'polarity']
j [label = 'relevant\nwords']
k [label = 'positivity\nlikelihood']
l [label = 'negativity\nlikelihood']

edge [color = grey]
a -> {b c d}
b -> {j}
c -> {e f g h i}
d -> {k l}
}")

text_schema %>%
  export_svg %>% charToRaw %>% rsvg_svg(file="./figures/text_schema.svg")

overall_schema <- grViz('digraph ninja{
  graph [compound = true, rankdir=LR, ranksep=1]

  node [fontcolor = black,
        shape = rectangle, width = 1.8, fixedsize = true,
        color = black, height=1, fontsize=25]

  edge [color = grey50, arrowtail = none]
  
  struct1 [
    label = "Twitter Dataset|<port1>IDs|<port2>Date|Flag|<port3>Text|<port4>User|<port5>Sentiment";
    shape=record
    fixedsize = false
    height=7.6
    fontsize=25
  ];
  struct1:port1 [style=filled, fillcolor=lightyellow]
  struct1:port1 -> g
  struct1:port2 -> b
  struct1:port3 -> c
  struct1:port4 -> c
  struct1:port5 -> d
  c -> d
  c -> e
  c -> f
  e -> g [color=transparent]
  e -> h [color=transparent]
  b -> h
  e -> i
  f -> j
  d -> k
  
  b [label="Feature\nExtraction", peripheries=2]
  c [label="Text\nCleaning", peripheries=2]
  d [label="Word\nEmbedding", peripheries=2]
  e [label="TF-DF", peripheries=2]
  f [label="Sentiment\nAnalysis", peripheries=2]
  g [label="Numerical\nValue", peripheries=1, style="filled", fillcolor= "#fdb462"]
  h [label="Discrete\nValue", peripheries=1, style="filled", fillcolor= "#fdb462"]
  i [label="Weighted\nMatrix", peripheries=1, style="filled", fillcolor= "#fdb462"]
  j [label="Sentiment\nFeatures", peripheries=1, style="filled", fillcolor= "#fdb462"]
  k [label="Likelihood\nFeatures", peripheries=1, style="filled", fillcolor= "#fdb462"]
  }')

overall_schema %>%
  export_svg %>% charToRaw %>% rsvg_svg(file="./figures/overall_schema.svg")
overall_schema