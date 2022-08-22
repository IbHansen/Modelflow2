Introduction
###################



**ModelFlow is written in Python**. Python comes "batteries included" and is
the basis of a very rich ecosystem, which consists of a wide array of
libraries. ModelFlow is just another library. It supplements the existing
libraries regarding modeling language and solving and allows the use of
Python as a model management framework.

**Data handling and wrangling is done in the Pandas library**. This
library is the Swiss army knife of data science in Python. It can import and export data to most systems and it is very powerful in manipulating and transforming data.
The core
element of Pandas is the *Dataframe*. A Dataframe is a two-dimensional
tabular data structure. Each *column* consists of cells of the same type
-- it can be a number, a string, a matrix or another Python data object.This includes matrices and other dataframes. Each *row is indexed.* The index can basically be any type of variable
including dates, which is especially relevant for economic and financial models.

**ModelFlow gives the user tools for more than solving models**. This
includes:

-   *Visualization* and comparison of results

-   *Integration* of models from different sources

-   *Analyze the logical structure of a model*. By applying graph theory, 
    ModelFlow can find data lineage, find a suitable calculating sequence and trace 
    causes of changes through the calculations.

-   *Inverting* the model to calculating the necessary instruments to
    achieve a desired target.

-   Calculating the *attributions* from input to the results of a model.

-   Calculating the *attribution* from input to the result of each
    formula.

-   Finding and calculating partial *derivatives* of formulas

-   *Integrating user defined python functions* in the Business logic
    language (like optimization, calculating risk weights or to make a matrices consistent with the RAS algorithm  )

-   *Wrap matlab* models so they can be used in the Business logic
    language.

-   *Speed up* solving using "Just in time compilation"

-   Analyze the model structure through tools from graph theory

-   Handle *large models.* 1,000,000 formulas is not a problem.

-   Integrate model management in Jupyter notebooks for *agile and user
    friendly model use*


**The core code of ModelFlow is small** Thus it can easily be modified and expanded to the specific need of the user. *ModelFlow is a toolset*. It can handle models, which conform to the tools.

If you need a feature or have a model which can't be handled in ModelFlow,
you are encouraged to improve ModelFlow. Please share the
improvement, other users may have the same need, or can be inspired by
your work.

Also bear in mind that ModelFlow is experimental. It is provided ”as is”, without any representation or warranty of any kind either express or implied.   
