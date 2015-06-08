//
//  AppDelegate.swift
//  PanoramaStitching
//
//  Created by Paul on 02/06/2015.
//  Copyright (c) 2015 Fluid Pixel. All rights reserved.
//

import Cocoa

let PanoramaStitchingApp = NSApplication.sharedApplication().delegate as? AppDelegate

@NSApplicationMain
class AppDelegate: NSObject, NSApplicationDelegate {

    var panoramaViewerWindow:NSWindowController!

    func applicationDidFinishLaunching(aNotification: NSNotification) {
        panoramaViewerWindow = NSStoryboard(name: "Main", bundle: nil)!.instantiateControllerWithIdentifier("PanoramaView") as! NSWindowController

        // Insert code here to initialize your application
    }

    func applicationWillTerminate(aNotification: NSNotification) {
        // Insert code here to tear down your application
    }

    
    func openPanoramaViewer(path:String) {
        let viewer = panoramaViewerWindow.contentViewController as! PanoramaViewer
        
        viewer.imagePath = path
        
        panoramaViewerWindow.showWindow(self)
        
    }
    
}
