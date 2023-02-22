/*
 To do:
  - Concept of the genetic algorithm.
    - A genetic algorithm is a problem-solving technique that imitates the process of natural selection. 
    - It starts with a population of potential solutions to a problem, and then selects the fittest ones to "reproduce" and create a new population of solutions.
    - Each solution is represented as a string of "genes", which are evaluated based on their fitness or how well they solve the problem. 
    - The fittest solutions are selected to "mate" or combine their genes to create new solutions, which are then evaluated again for their fitness.
    - This process is repeated over multiple generations, with each generation improving upon the previous one as the fittest solutions are selected and reproduced. 
    - Eventually, the genetic algorithm converges on the best solution to the problem, which is the one with the highest fitness in the final generation. 
  - Modify your code to implement the genetic algorithm instead of a random search.
  - Choose the initial population of individuals by creating random expressions.
  - Evaluate the fitness of each individual in the population by comparing their output to a set of sample data.
  - Set a termination condition for the algorithm, such as a time limit or a sufficient fitness level achieved.
  - Select the best-ranking individuals in the population to reproduce.
  - Breed the new generation of individuals through crossover and/or mutation to create offspring.
  - Evaluate the individual fitnesses of the offspring.
  - Replace the lower-ranked part of the population with the offspring.
  - Keep track of the best expression found so far and output the generation number, RMS error, and expression every time a new best expression is found.
  - Compare the performance of your genetic algorithm to the random search algorithm by allowing both to run for 60 seconds and keeping track of the best RMS error and expression for each.
  - Use the old Genetic Algorithms demonstration applet to get a better understanding of how genetic algorithms work.
  - Consider using LISP programming language and tree structures to represent and manipulate the expressions as programs for genetic programming.
  - Experiment with different tuning options and decisions for the genetic algorithm to see how it affects the results.
  - Remember that this is an experimental lab and the results may vary depending on the details of your implementation.
 */
import java.util.*;

public class GeneticProgramming {
    public static void main(String[] args) {
        // generate some random expressions of different heights
        int numExpressions = 5;
        int maxHeight = 3;
        List<ExpNode> expressions = new ArrayList<>();
        for (int i = 0; i < numExpressions; i++) {
            expressions.add(randomExpression(maxHeight));
        }
        // print the expressions
        for (ExpNode exp : expressions) {
            System.out.println(exp.toString());
        }
    }

    //node classes
    static abstract class ExpNode {
        abstract double value(double x);
        abstract ExpNode copy();
    }

    static class ConstNode extends ExpNode {
        double value;
        ConstNode(double val) {
            value = val;
        }
        double value(double x) {
            return value;
        }
        ExpNode copy() {
            return new ConstNode(value);
        }
        public String toString() {
            return String.format("%.2f", value);
        }
    }

    static class VariableNode extends ExpNode {
        double value(double x) {
            return x;
        }
        ExpNode copy() {
            return new VariableNode();
        }
        public String toString() {
            return "x";
        }
    }

    static class BinaryOpNode extends ExpNode {
        char operator;
        ExpNode left, right;
        BinaryOpNode(char op, ExpNode l, ExpNode r) {
            operator = op;
            left = l;
            right = r;
        }
        double value(double x) {
            double leftVal = left.value(x);
            double rightVal = right.value(x);
            switch (operator) {
                case '+':
                    return leftVal + rightVal;
                case '-':
                    return leftVal - rightVal;
                case '*':
                    return leftVal * rightVal;
                case '/':
                    return leftVal / rightVal;
                case '^':
                    return Math.pow(leftVal, rightVal);
                default:
                    return Double.NaN;
            }
        }
        ExpNode copy() {
            return new BinaryOpNode(operator, left.copy(), right.copy());
        }
        public String toString() {
            String leftStr = left instanceof BinaryOpNode ? "(" + left.toString() + ")" : left.toString();
            String rightStr = right instanceof BinaryOpNode ? "(" + right.toString() + ")" : right.toString();
            return leftStr + " " + operator + " " + rightStr;
        }
    }

    static class UnaryOpNode extends ExpNode {
        int function;
        ExpNode child;

        UnaryOpNode(int func, ExpNode c) {
            function = func;
            child = c;
        }

        double value(double x) {
            double childVal = child.value(x);
            switch (function) {
                case 0:
                    return Math.sin(childVal);
                case 1:
                    return Math.cos(childVal);
                case 2:
                    return Math.exp(childVal);
                case 3:
                    return Math.abs(childVal);
                case 4:
                    return -childVal;
                default:
                    return Double.NaN;
            }
        }

        ExpNode copy() {
            return new UnaryOpNode(function, child.copy());
        }

        public String toString() {
            String childStr = child instanceof BinaryOpNode ? "(" + child.toString() + ")" : child.toString();
            switch (function) {
                case 0:
                    return "sin(" + childStr + ")";
                case 1:
                    return "cos(" + childStr + ")";
                case 2:
                    return "exp(" + childStr + ")";
                case 3:
                    return "abs(" + childStr + ")";
                case 4:
                    return "-" + childStr;
                default:
                    return "NaN";
            }
        }
    }
  

    // generate a random expression tree of a given height
    static ExpNode randomExpression(int height) {
        if (height == 0) {
            // base case: return a random constant or variable
            if (Math.random() < 0.5) {
                return new ConstNode(Math.random() * 10 - 5);
            } else {
                return new VariableNode();
            }
        } else {
            // recursive case: return a random binary operator with random children
            char[] operators = {'+', '-', '*', '/', '^'};
            char op = operators[(int) (Math.random() * operators.length)];
            ExpNode left = randomExpression(height - 1);
            ExpNode right = randomExpression(height - 1);
            return new BinaryOpNode(op, left, right);
        }
    }
}