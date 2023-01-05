Fake News Detection with R
================

-   [About the project](#about-the-project)
    -   [What is NLP](#what-is-nlp)
-   [Data Loading](#data-loading)
-   [Distribution fake/true news](#distribution-faketrue-news)
-   [Which autor are more associated with fake/true
    news?](#which-autor-are-more-associated-with-faketrue-news)
-   [Which words are more associated with fake/true
    news?](#which-words-are-more-associated-with-faketrue-news)
-   [Are fake news lengthy than the true
    ones?](#are-fake-news-lengthy-than-the-true-ones)
-   [Model creation](#model-creation)
    -   [Data Splitting](#data-splitting)
    -   [Preprocessing the text](#preprocessing-the-text)
    -   [Defining the models](#defining-the-models)

## About the project

### What is NLP

## Data Loading

``` r
library(tidymodels)
```

    ## Warning: package 'tidymodels' was built under R version 4.1.3

    ## -- Attaching packages -------------------------------------- tidymodels 1.0.0 --

    ## v broom        1.0.1      v recipes      1.0.3 
    ## v dials        1.1.0      v rsample      1.1.1 
    ## v dplyr        1.0.10     v tibble       3.1.8 
    ## v ggplot2      3.4.0      v tidyr        1.2.1 
    ## v infer        1.0.4      v tune         1.0.1 
    ## v modeldata    1.0.1      v workflows    1.1.2 
    ## v parsnip      1.0.3      v workflowsets 1.0.0 
    ## v purrr        0.3.5      v yardstick    1.1.0

    ## Warning: package 'broom' was built under R version 4.1.3

    ## Warning: package 'dials' was built under R version 4.1.3

    ## Warning: package 'scales' was built under R version 4.1.3

    ## Warning: package 'dplyr' was built under R version 4.1.3

    ## Warning: package 'ggplot2' was built under R version 4.1.3

    ## Warning: package 'infer' was built under R version 4.1.3

    ## Warning: package 'modeldata' was built under R version 4.1.3

    ## Warning: package 'parsnip' was built under R version 4.1.3

    ## Warning: package 'purrr' was built under R version 4.1.3

    ## Warning: package 'recipes' was built under R version 4.1.3

    ## Warning: package 'rsample' was built under R version 4.1.3

    ## Warning: package 'tibble' was built under R version 4.1.3

    ## Warning: package 'tidyr' was built under R version 4.1.3

    ## Warning: package 'tune' was built under R version 4.1.3

    ## Warning: package 'workflows' was built under R version 4.1.3

    ## Warning: package 'workflowsets' was built under R version 4.1.3

    ## Warning: package 'yardstick' was built under R version 4.1.3

    ## -- Conflicts ----------------------------------------- tidymodels_conflicts() --
    ## x purrr::discard() masks scales::discard()
    ## x dplyr::filter()  masks stats::filter()
    ## x dplyr::lag()     masks stats::lag()
    ## x recipes::step()  masks stats::step()
    ## * Search for functions across packages at https://www.tidymodels.org/find/

``` r
library(tidytext)
```

    ## Warning: package 'tidytext' was built under R version 4.1.3

``` r
library(ranger)
```

    ## Warning: package 'ranger' was built under R version 4.1.3

``` r
library(textrecipes)
```

    ## Warning: package 'textrecipes' was built under R version 4.1.3

``` r
library(here)
```

    ## Warning: package 'here' was built under R version 4.1.1

    ## here() starts at C:/Users/pessoal/Desktop/Projetos/nlp_pipeline_optimization

``` r
library(tidyverse)
```

    ## Warning: package 'tidyverse' was built under R version 4.1.3

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.2 --

    ## v readr   2.1.3     v forcats 0.5.2
    ## v stringr 1.5.0

    ## Warning: package 'readr' was built under R version 4.1.3

    ## Warning: package 'stringr' was built under R version 4.1.3

    ## Warning: package 'forcats' was built under R version 4.1.3

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x readr::col_factor() masks scales::col_factor()
    ## x purrr::discard()    masks scales::discard()
    ## x dplyr::filter()     masks stats::filter()
    ## x stringr::fixed()    masks recipes::fixed()
    ## x dplyr::lag()        masks stats::lag()
    ## x readr::spec()       masks yardstick::spec()

``` r
news <- read_csv(paste0(here(), "/Data/train.csv"))
```

    ## Rows: 20800 Columns: 5
    ## -- Column specification --------------------------------------------------------
    ## Delimiter: ","
    ## chr (3): title, author, text
    ## dbl (2): id, label
    ## 
    ## i Use `spec()` to retrieve the full column specification for this data.
    ## i Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
glimpse(news)
```

    ## Rows: 20,800
    ## Columns: 5
    ## $ id     <dbl> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1~
    ## $ title  <chr> "House Dem Aide: We Didn’t Even See Comey’s Letter Until Jason ~
    ## $ author <chr> "Darrell Lucus", "Daniel J. Flynn", "Consortiumnews.com", "Jess~
    ## $ text   <chr> "House Dem Aide: We Didn’t Even See Comey’s Letter Until Jason ~
    ## $ label  <dbl> 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, ~

``` r
news <- news %>% 
  mutate(content = paste(title, text, sep = " "),
         label = if_else(label == 1, "true", "fake"),
         label = factor(label, levels = c("true","fake"))) %>% # Setting 'true' as first factor 
  relocate(label, .before = 1) %>% 
  select(-title,-text) 
news
```

    ## # A tibble: 20,800 x 4
    ##    label    id author                       content                             
    ##    <fct> <dbl> <chr>                        <chr>                               
    ##  1 true      0 Darrell Lucus                "House Dem Aide: We Didn’t Even See~
    ##  2 fake      1 Daniel J. Flynn              "FLYNN: Hillary Clinton, Big Woman ~
    ##  3 true      2 Consortiumnews.com           "Why the Truth Might Get You Fired ~
    ##  4 true      3 Jessica Purkiss              "15 Civilians Killed In Single US A~
    ##  5 true      4 Howard Portnoy               "Iranian woman jailed for fictional~
    ##  6 fake      5 Daniel Nussbaum              "Jackie Mason: Hollywood Would Love~
    ##  7 true      6 nan                          "Life: Life Of Luxury: Elton John’s~
    ##  8 fake      7 Alissa J. Rubin              "Benoît Hamon Wins French Socialist~
    ##  9 fake      8 nan                          "Excerpts From a Draft Script for D~
    ## 10 fake      9 Megan Twohey and Scott Shane "A Back-Channel Plan for Ukraine an~
    ## # ... with 20,790 more rows

## Distribution fake/true news

``` r
news %>% count(label)
```

    ## # A tibble: 2 x 2
    ##   label     n
    ##   <fct> <int>
    ## 1 true  10413
    ## 2 fake  10387

-   Dataset seems quite balanced

## Which autor are more associated with fake/true news?

``` r
news %>% 
  filter(!str_detect(author, "^\\d|-|[:space:]|[:blank:]")) %>% 
  count(label, author)
```

    ## # A tibble: 560 x 3
    ##    label author            n
    ##    <fct> <chr>         <int>
    ##  1 true  AARGH63           1
    ##  2 true  abinico           1
    ##  3 true  Abramo            1
    ##  4 true  ActivistPost      8
    ##  5 true  admin           193
    ##  6 true  Admin            41
    ##  7 true  administrator     1
    ##  8 true  adobochron        1
    ##  9 true  Adoriasoft        1
    ## 10 true  AFP               1
    ## # ... with 550 more rows

``` r
  # filter(n > 10, label == "fake")
  # ggplot(aes(author, n, fill = label)) + 
  # geom_col() +
  # coord_flip()
```

## Which words are more associated with fake/true news?

``` r
## Try different approaches: term-frequency, tf-idf, weighted log-odds (tidylo package)
news %>% 
  unnest_tokens(text, content)
```

    ## # A tibble: 15,963,949 x 4
    ##    label    id author        text   
    ##    <fct> <dbl> <chr>         <chr>  
    ##  1 true      0 Darrell Lucus house  
    ##  2 true      0 Darrell Lucus dem    
    ##  3 true      0 Darrell Lucus aide   
    ##  4 true      0 Darrell Lucus we     
    ##  5 true      0 Darrell Lucus didn’t 
    ##  6 true      0 Darrell Lucus even   
    ##  7 true      0 Darrell Lucus see    
    ##  8 true      0 Darrell Lucus comey’s
    ##  9 true      0 Darrell Lucus letter 
    ## 10 true      0 Darrell Lucus until  
    ## # ... with 15,963,939 more rows

## Are fake news lengthy than the true ones?

## Model creation

### Data Splitting

### Preprocessing the text

### Defining the models
