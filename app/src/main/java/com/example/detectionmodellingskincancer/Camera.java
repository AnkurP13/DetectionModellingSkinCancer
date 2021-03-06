package com.example.detectionmodellingskincancer;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;

import com.google.android.material.bottomnavigation.BottomNavigationView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.provider.MediaStore;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

public class Camera extends AppCompatActivity {

    ImageView imageView2;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        //intialize and assign variables
        BottomNavigationView bottomNavigationView = findViewById(R.id.bottom_navigation);

        //set camera selected page
        bottomNavigationView.setSelectedItemId(R.id.Cam);

        //perform itemselcetedListener
        bottomNavigationView.setOnNavigationItemSelectedListener(new BottomNavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
                switch (menuItem.getItemId()) {
                    case R.id.map:
                        startActivity(new Intent(getApplicationContext(), MapsActivity.class));
                        overridePendingTransition(0, 0);
                        return true;
                    case R.id.Info:
                        startActivity(new Intent(getApplicationContext(), Articles.class));
                        overridePendingTransition(0, 0);
                        return true;
                    case R.id.Cam:
                        startActivity(new Intent(getApplicationContext(), Camera.class));
                        overridePendingTransition(0, 0);
                        return true;
                }
                return false;
            }
        });



        Button btncam = (Button)(findViewById(R.id.btncam));
        imageView2 = (ImageView)(findViewById(R.id.imageView2));


        btncam.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 0);

            }
        });


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Bitmap bitmap = (Bitmap)data.getExtras().get("data");
        imageView2.setImageBitmap(bitmap);
    }


    public void Location(View v){
        Intent myIntent = new Intent(getBaseContext(),   MapsActivity.class);
        startActivity(myIntent);
    }

    public void Info(View v){
        Intent myIntent = new Intent(getBaseContext(),   Articles.class);
        startActivity(myIntent);
    }
}