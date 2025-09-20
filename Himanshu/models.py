from dataclasses import dataclass, field
from typing import List, Dict, Set
from datetime import datetime
from collections import defaultdict
import numpy as np

class Person:
    def __init__(self, id: int, bbox: List[float], first_seen: float):
        self.id = id
        self.bboxes = [bbox]
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.total_time = 0
        self.areas_visited = []
        self.area_times = {}
        self.current_area = None
        self.last_area_entry = None
        
    def update_position(self, bbox: List[float], timestamp: float, area: str):
        self.bboxes.append(bbox)
        self.last_seen = timestamp
        
        if area != self.current_area:
            if self.current_area:
                # Update time spent in previous area
                time_in_area = timestamp - self.last_area_entry
                self.area_times[self.current_area] = self.area_times.get(self.current_area, 0) + time_in_area
            
            self.current_area = area
            self.last_area_entry = timestamp
            if area not in self.areas_visited:
                self.areas_visited.append(area)
    
    def calculate_total_time(self):
        self.total_time = self.last_seen - self.first_seen
        return self.total_time
    
    def determine_shopping_behavior(self, min_areas: int = 2, min_time: float = 30.0) -> bool:
        """Determine if person was shopping based on areas visited and time spent"""
        unique_areas = len(self.areas_visited)
        total_time = self.calculate_total_time()
        return unique_areas >= min_areas and total_time >= min_time

@dataclass
class AreaStats:
    name: str
    visit_count: int = 0
    total_dwell_time: float = 0.0
    peak_visitors: int = 0
    peak_time: datetime = None
    current_visitors: int = 0
    visitor_history: List[int] = field(default_factory=list)
    
    def update(self, visitors: int, timestamp: datetime):
        self.current_visitors = visitors
        self.visitor_history.append(visitors)
        
        if visitors > self.peak_visitors:
            self.peak_visitors = visitors
            self.peak_time = timestamp
    
    def add_visit(self, dwell_time: float):
        self.visit_count += 1
        self.total_dwell_time += dwell_time
    
    @property
    def average_dwell_time(self) -> float:
        return self.total_dwell_time / self.visit_count if self.visit_count > 0 else 0.0
        
        if current_area:
            if self.last_area != current_area:
                self.shopping_areas_visited.add(current_area)
                self.last_area = current_area
            
            # Update dwell time
            if len(self.timestamps) > 1:
                time_diff = timestamp - self.timestamps[-2]
                self.area_dwell_times[current_area] += time_diff
                
    def calculate_total_time(self) -> float:
        """Calculate total time spent in mall"""
        if len(self.timestamps) > 1:
            self.total_time_in_mall = self.timestamps[-1] - self.timestamps[0]
        return self.total_time_in_mall
    
    def determine_shopping_behavior(self, min_areas_for_shopping: int = 2, 
                                  min_dwell_time: float = 30.0) -> bool:
        """Determine if person went shopping based on areas visited and dwell time"""
        total_dwell = sum(self.area_dwell_times.values())
        areas_with_significant_time = sum(1 for time in self.area_dwell_times.values() 
                                        if time > min_dwell_time)
        
        self.is_shopping = (len(self.shopping_areas_visited) >= min_areas_for_shopping and
                          areas_with_significant_time >= 1 and
                          total_dwell > 60.0)  # At least 1 minute total dwell time
        return self.is_shopping

@dataclass
class AreaStats:
    name: str
    visit_count: int = 0
    total_dwell_time: float = 0.0
    peak_visitors: int = 0
    peak_time: datetime = None
    current_visitors: int = 0
    visitor_history: List[int] = field(default_factory=list)
    unique_visitors: Set[int] = field(default_factory=set)
    
    def update(self, visitors: int, timestamp: datetime):
        self.current_visitors = visitors
        self.visitor_history.append(visitors)
        
        if visitors > self.peak_visitors:
            self.peak_visitors = visitors
            self.peak_time = timestamp
    
    def add_visit(self, dwell_time: float):
        self.visit_count += 1
        self.total_dwell_time += dwell_time
    
    @property
    def average_dwell_time(self) -> float:
        return self.total_dwell_time / self.visit_count if self.visit_count > 0 else 0.0
