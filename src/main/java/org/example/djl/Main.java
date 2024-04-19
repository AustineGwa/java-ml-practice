package org.example.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;



public class Main {

    public static double elapsedTime(long start) {
        long now = System.currentTimeMillis();
        return (now - start) / 1000.0;
    }

    public static void main(String[] args) {

        //compare djl vector computations and java arrays
        compareDjlToArray();


    }

    private static void compareDjlToArray() {
        //instantiate two  10_000-dimensional vectors containing all ones
        int n = 10_000;
        NDManager manager = NDManager.newBaseManager();
        NDArray a = manager.ones(new Shape(n));
        NDArray b = manager.ones(new Shape(n));

        //using a for loop
        NDArray c = manager.zeros(new Shape(n));
        long start1 = System.currentTimeMillis();
        for (int i = 0; i < n; i++) {
            c.set(new NDIndex(i), a.getFloat(i) + b.getFloat(i));
        }

        System.out.println("C: " +c);

        System.out.println("Time taken for loop "+String.format("%.5f sec", elapsedTime(start1)));


        //Alternatively,relying on DJL to compute the elementwise sum:
        long start2 = System.currentTimeMillis();
        NDArray d = a.add(b);
        System.out.println( "Time taken DJL " +String.format("%.5f sec", elapsedTime(start2)));
    }
}
