package com.actitracker.data;

/**
 * Created by jvalantpatel on 11/26/15.
 */
public class Constants {

    public final static Long interval = 5000L;
    public final static Long jump = 300000L;
    public final static double training = 0.6;
    public final static double test = 0.4;

    public static void printConstants(){
        System.out.println("interval -"+interval+" jump -"+jump+" training - "
                +training+" test - "+test);
    }

}
