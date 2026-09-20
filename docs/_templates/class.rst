{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
   :no-members:
   :no-inherited-members:

   {% set native_methods = skactiveml_methods.get(fullname, ()) %}
   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
   {%- if item != '__init__' and item in native_methods %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% for item in methods %}
   {%- if item != '__init__' and item not in native_methods %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% for item in methods %}
{%- if item != '__init__' and item in native_methods %}
.. automethod:: {{ fullname }}.{{ item }}
{%- endif %}
{%- endfor %}
{% for item in methods %}
{%- if item != '__init__' and item not in native_methods %}
.. automethod:: {{ fullname }}.{{ item }}
{%- endif %}
{%- endfor %}
{% for item in attributes %}
.. autoattribute:: {{ fullname }}.{{ item }}
{%- endfor %}

.. _sphx_glr_backref_{{fullname}}:

.. minigallery:: {{fullname}}
   :add-heading: Examples Using ``{{ fullname }}``
