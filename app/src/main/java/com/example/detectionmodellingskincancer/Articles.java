package com.example.detectionmodellingskincancer;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

public class Articles extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_articles);

        //intialize and assign variables
        BottomNavigationView bottomNavigationView = findViewById(R.id.bottom_navigation);

        //set camera selected page
        bottomNavigationView.setSelectedItemId(R.id.info);

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


    }


    public void skinvision(View view){
        Intent skinIntent = new Intent(Intent.ACTION_VIEW, Uri.parse("https://siliconcanals.com/news/skinvision-ai-app-skin-cancer-amsterdam-startup/"));
        startActivity(skinIntent);

    }

    public void treatone(View view){
        Intent t1intent = new Intent(Intent.ACTION_VIEW, Uri.parse("https://www.curetoday.com/publications/cure/2020/immunotherapy-2020/speaking-out-taking-action-against-skin-cancer"));
        startActivity(t1intent);

    }

    public void treattwo(View view){
        Intent t2intent = new Intent(Intent.ACTION_VIEW, Uri.parse("https://www.healtheuropa.eu/potential-new-treatment-for-skin-cancer-with-wearable-patch-development/100696/"));
        startActivity(t2intent);

    }
    public void healthc(View view){
        Intent healt = new Intent(Intent.ACTION_VIEW, Uri.parse("https://www.everydayhealth.com/cancer/skin-cancer/can-essential-oils-work-for-skin-cancer/"));
        startActivity(healt);

    }


}