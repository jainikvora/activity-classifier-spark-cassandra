package com.actitracker.data;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.List;
import java.util.logging.Logger;

/**
 * Created by jainikkumar on 11/24/15.
 */
public class SampledDataDump {

    private static Connection getConnection() {
        Connection conn = null;
        try{
             conn = DriverManager.getConnection("jdbc:mysql://localhost/actitracker", "root", "admin");
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
        return conn;
    }

    public static void saveDataToMySQL(List<List<Double>> sampleList) {
        Connection conn = getConnection();
        String query = "INSERT INTO activity_with_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        try {
            for(List<Double> sample: sampleList) {
                PreparedStatement stmt = conn.prepareStatement(query);
                for(int i=1; i <= sample.size(); i++) {
                    stmt.setDouble(i, sample.get(i-1));
                }
                stmt.execute();
            }
            conn.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }
}
