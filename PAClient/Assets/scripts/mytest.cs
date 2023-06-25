using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using fts;

// Define the delegate type that matches the function signature in the C++ code
public delegate int AddDelegate(int a, int b);

// Define a class to hold the delegate instance
[PluginAttr("libPAMLDLL")]
public class MyPlugin
{
    [PluginFunctionAttr("Add")]
    public static AddDelegate Add;
}


public class mytest : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log(MyPlugin.Add(4, 5) + "a");
    }

    // Update is called once per frame
    void Update()
    {

    }
}
