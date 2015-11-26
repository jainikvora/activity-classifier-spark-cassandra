package com.actitracker.data;

import org.junit.Test;

import java.util.*;

/**
 * Created by jainikkumar on 11/24/15.
 */
public class SampledDataDumpTest {
    @Test
    public void testSaveDataToMySQL() throws Exception {
        Double[] feature = {1.0, 3.3809183673469394, -6.880102040816324, 0.8790816326530612, 50.08965378708187, 84.13105050494424, 20.304453787081833, 5.930491461890875, 7.544194085797583, 3.519248229904206, 12.968485972481643, 7.50031E8};
        Double[] feature1 = {1.0, 3.3809183673469394, -6.880102040816324, 0.8790816326530612, 50.08965378708187, 84.13105050494424, 20.304453787081833, 5.930491461890875, 7.544194085797583, 3.519248229904206, 12.968485972481643, 7.50031E8};
        List<List<Double>> testList = new ArrayList<>();
        testList.add(new ArrayList<>(Arrays.asList(feature)));
        testList.add(new ArrayList<>(Arrays.asList(feature1)));
        SampledDataDump.saveDataToMySQL(testList);

    }
}
