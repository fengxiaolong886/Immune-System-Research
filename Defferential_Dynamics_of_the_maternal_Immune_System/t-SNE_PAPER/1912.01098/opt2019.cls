\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{opt2019}[2019/02/01 v1.0 OPT 2019 Style File]

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[wcp]{jmlr}

% modified so that header doesn't appear
\jmlrvolume{}
\jmlryear{}
\jmlrproceedings{wcp}{OPT2019: 11th Annual Workshop on Optimization for Machine Learning}


\ifanonsubmission
 \newcommand{\optauthor}[1]{}
 \author{author names withheld}
 \editor{Under Review for OPT 2019}
 \jmlrworkshop{11th Annual Workshop on Optimization for Machine Learning}
 \renewcommand{\acks}[1]{}
\else
 \newcommand{\optauthor}[1]{\author{#1}}
 \iffinalsubmission
  \editors{}
  \jmlrworkshop{11th Annual Workshop on Optimization for Machine Learning}
 \fi
\fi





