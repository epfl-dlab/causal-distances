#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class CausalModel(ABC):
    @abstractmethod
    def intervene(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def nodes(self):
        pass
